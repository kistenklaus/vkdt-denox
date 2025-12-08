#include "compute_graph.hpp"
#include "compress_weights.hpp"
#include <algorithm>
#include <dnx.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <stdexcept>

vkdt_denox::ComputeGraph vkdt_denox::reconstruct_compute_graph(
    const denox::dnx::Model *dnx, const CompressedWeights &compressed_weights) {
  const uint32_t buffer_count = dnx->buffers()->size();
  const uint32_t tensor_count = dnx->tensors()->size();
  const uint32_t dispatch_count = dnx->dispatches()->size();

  // maps buffer ids to owning nodes.
  struct BufferLocation {
    uint32_t owning_node;
    uint32_t borrowing_node;
    uint32_t sinksource_id;
    uint32_t buffer_roi_id;
  };
  std::vector<BufferLocation> buffer_locations( //
      buffer_count,                             //
      BufferLocation{
          .owning_node = none_sentinal,
          .borrowing_node = none_sentinal,
          .sinksource_id = 0,
          .buffer_roi_id = none_sentinal,
      });

  // Create weight node.
  ComputeGraph graph;
  uint32_t weight_buffer_roi_id = graph.buffer_rois.size();
  graph.buffer_rois.push_back(BufferRoi{
      .byte_size = compressed_weights.data.size(),
  });
  size_t weight_node_id = graph.nodes.size();
  graph.nodes.push_back(Node{
      .op =
          Upload{
              .name = "weights",
              .sinksource_id = 0,
          },
      .sinksources = {SinkSource{
          .name = "weights",
          .type = SinkSourceType::Source,
          .chan = SinkSourceChan::SSBO,
          .format = SinkSourceFormat::Byte,
          .buffer_roi_id = weight_buffer_roi_id,
      }},
  });

  const uint32_t initalizer_count = dnx->initializers()->size();
  for (uint32_t i = 0; i < initalizer_count; ++i) {
    const auto *initalizer = dnx->initializers()->Get(i);
    const uint32_t tensor_id = initalizer->tensor();
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const uint32_t buffer_id = tensor->buffer();
    buffer_locations[buffer_id].owning_node = weight_node_id;
    buffer_locations[buffer_id].sinksource_id = 0;
    buffer_locations[buffer_id].borrowing_node = none_sentinal;
    buffer_locations[buffer_id].buffer_roi_id = weight_buffer_roi_id;
  }

  // Write rois for input!

  fmt::println("");
  for (uint32_t i = 0; i < dnx->inputs()->size(); ++i) {
    const auto *tensor_info = dnx->inputs()->Get(i);
    const uint32_t tensor_id = tensor_info->tensor();
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const uint32_t buffer_id = tensor->buffer();
    const auto *buffer = dnx->buffers()->Get(buffer_id);
    const uint32_t input_roi_id = graph.buffer_rois.size();
    // TODO: Change to extent and type based semantics.
    graph.buffer_rois.push_back(BufferRoi{
        .byte_size =
            Symbol{
                .type = buffer->size_type(),
                .ptr = buffer->size(),
            },
    });

    buffer_locations[buffer_id].owning_node = input_sential;
    buffer_locations[buffer_id].sinksource_id = i;
    buffer_locations[buffer_id].borrowing_node = none_sentinal;
    buffer_locations[buffer_id].buffer_roi_id = input_roi_id;
  }

  for (uint32_t d = 0; d < dispatch_count; ++d) {
    const auto dispatch_type = dnx->dispatches_type()->Get(d);
    if (dispatch_type == denox::dnx::Dispatch_ComputeDispatch) {
      uint32_t node_id = graph.nodes.size();
      const auto *compute_dispatch =
          dnx->dispatches()->GetAs<denox::dnx::ComputeDispatch>(d);

      struct TensorBinding {
        uint16_t set;
        uint16_t binding;
        uint8_t access;
        uint32_t buffer;
        Symbol offset;
      };
      std::vector<TensorBinding> bindings;

      const uint32_t binding_count = compute_dispatch->bindings()->size();
      for (uint32_t b = 0; b < binding_count; ++b) {
        const auto *binding = compute_dispatch->bindings()->Get(b);
        const uint32_t tensor_binding_count = binding->bindings()->size();
        for (uint32_t t = 0; t < tensor_binding_count; ++t) {
          const auto *tensor_binding = binding->bindings()->Get(t);
          const uint32_t tensor_id = tensor_binding->tensor();
          const auto *tensor = dnx->tensors()->Get(tensor_id);
          const uint32_t buffer_id = tensor->buffer();

          bindings.push_back(TensorBinding{
              .set = binding->set(),
              .binding = tensor_binding->binding(),
              .access = tensor_binding->access(),
              .buffer = buffer_id,
              .offset =
                  Symbol{
                      .type = tensor->offset_type(),
                      .ptr = tensor->offset(),
                  },
          });
        }
      }
      std::sort(bindings.begin(), bindings.end(),
                [](const auto &lhs, const auto &rhs) {
                  if (lhs.set < rhs.set) {
                    return true;
                  } else if (lhs.set > rhs.set) {
                    return false;
                  } else {
                    return lhs.binding < rhs.binding;
                  }
                });
      // TODO: Enforce vkdt binding semantics.
      // - Linear descriptor bindings, starting at binding 0.
      // - Only use set 1.

      std::vector<SinkSource> sinksources;
      uint32_t dummy_sink_id = bindings.size();
      // TODO: Make sure that sinks are actually created.
      sinksources.reserve(bindings.size());
      for (uint32_t b = 0; b < bindings.size(); ++b) {
        uint32_t sinksource_id = b;
        const auto &binding = bindings[b];
        const auto *buffer = dnx->buffers()->Get(binding.buffer);

        auto &location = buffer_locations[binding.buffer];

        SinkSourceType type;
        if (binding.access == denox::dnx::Access_WriteOnly) {
          if (location.owning_node != none_sentinal) {
            // Some other nodes has already written to this node.
            // We apply the following rule:
            // Let A be the owning node (i.e. the node that initally wrote)
            // Let B be the current node (i.e. node_id)
            // Add RAW edge between A and B
            //   (This is slightly inaccurate WAW would be correct, but RAW wil
            //   lead to the same synchronization mechanism on most devices).
            //
            // If borrowing node is unequal to sential, then let this node be C.
            // Add dummy write ssbo to C.
            // Add RAW edge between C and B.
            // This ensures that A -> C -> B
            // Finally update borrowing_node.
            type = SinkSourceType::Read;
            graph.connectors.push_back(Connector{
                .src_node = location.owning_node,
                .src_node_sinksource = location.sinksource_id,
                .dst_node = node_id,
                .dst_node_sinksource = sinksource_id,
            });

            if (location.borrowing_node != none_sentinal) {
              // insert dummy edge
              auto &node_c = graph.nodes[location.borrowing_node];
              if (!node_c.dummy_source.has_value()) {
                if (!graph.dummy_roi.has_value()) {
                  graph.dummy_roi = graph.buffer_rois.size();
                  graph.buffer_rois.push_back(BufferRoi{
                      .byte_size = 1ull, // <- possibly to small
                  });
                }
                const uint32_t dummy_roi = graph.dummy_roi.value();
                node_c.dummy_source = node_c.sinksources.size();
                node_c.sinksources.push_back(SinkSource{
                    .name = "dummy-source",
                    .type = SinkSourceType::Write,
                    .chan = SinkSourceChan::SSBO,
                    .format = SinkSourceFormat::Byte,
                    .buffer_roi_id = dummy_roi,
                });
              }
              const uint32_t dummy_source = node_c.dummy_source.value();
              const uint32_t dummy_sink = dummy_sink_id++;

              graph.connectors.push_back(Connector{
                  .src_node = location.borrowing_node,
                  .src_node_sinksource = dummy_source,
                  .dst_node = node_id,
                  .dst_node_sinksource = dummy_sink,
              });
            }
            location.borrowing_node = node_id;
          } else {
            assert(location.owning_node == none_sentinal);
            assert(location.buffer_roi_id == none_sentinal);
            uint32_t buffer_roi_id = graph.buffer_rois.size();
            graph.buffer_rois.push_back(BufferRoi{
                .byte_size =
                    Symbol{
                        .type = buffer->size_type(),
                        .ptr = buffer->size(),
                    },
            });
            location.owning_node = node_id;
            location.buffer_roi_id = buffer_roi_id;
            location.borrowing_node = none_sentinal;
            location.sinksource_id = sinksource_id;
            type = SinkSourceType::Write; // <- allocates resource
          }
          assert(location.buffer_roi_id != none_sentinal);
        } else if (binding.access == denox::dnx::Access_ReadOnly) {
          assert(location.owning_node != none_sentinal);
          assert(location.buffer_roi_id != none_sentinal);
          graph.connectors.push_back(Connector{
              .src_node = location.owning_node,
              .src_node_sinksource = location.sinksource_id,
              .dst_node = node_id,
              .dst_node_sinksource = sinksource_id,
          });
          if (location.borrowing_node != none_sentinal) {
            // insert dummy edge.
            auto &node_c = graph.nodes[location.borrowing_node];
            if (!node_c.dummy_source.has_value()) {
              if (!graph.dummy_roi.has_value()) {
                graph.dummy_roi = graph.buffer_rois.size();
                graph.buffer_rois.push_back(BufferRoi{
                    .byte_size = 1ull,
                });
              }
              const uint32_t dummy_roi = graph.dummy_roi.value();

              node_c.dummy_source = node_c.sinksources.size();
              node_c.sinksources.push_back(SinkSource{
                  .name = "dummy-source",
                  .type = SinkSourceType::Write,
                  .chan = SinkSourceChan::SSBO,
                  .format = SinkSourceFormat::Byte,
                  .buffer_roi_id = dummy_roi,
              });
            }
            const uint32_t dummy_source = node_c.dummy_source.value();
            const uint32_t dummy_sink = dummy_sink_id++;

            graph.connectors.push_back(Connector{
                .src_node = location.borrowing_node,
                .src_node_sinksource = dummy_source,
                .dst_node = node_id,
                .dst_node_sinksource = dummy_sink,
            });
          }

          type = SinkSourceType::Read;
        } else if (binding.access == denox::dnx::Access_ReadWrite) {
          throw std::runtime_error(
              "vkdt_denox: readwrite access is not supported!");
        } else {
          throw std::runtime_error("invalid tensor binding access");
        }
        assert(location.owning_node != none_sentinal);
        SinkSourceChan chan = SinkSourceChan::SSBO;
        // TODO: Set appropriate format and roi for in / output tensors.
        SinkSourceFormat format = SinkSourceFormat::Byte;

        sinksources.push_back(
            SinkSource{.name = fmt::format("b{}", b),
                       .type = type,
                       .chan = chan,
                       .format = format,
                       .buffer_roi_id = location.buffer_roi_id});
      }

      uint32_t dummy_sink_count = dummy_sink_id - bindings.size();

      if (dummy_sink_count != 0 && !graph.dummy_roi.has_value()) {
        graph.dummy_roi = graph.buffer_rois.size();
        graph.buffer_rois.push_back(BufferRoi{
            .byte_size = 1ull,
        });
      }

      for (uint32_t i = 0; i < dummy_sink_count; ++i) {

        sinksources.push_back(SinkSource{
            .name = fmt::format("dummy-sink-{}", i),
            .type = SinkSourceType::Read,
            .chan = SinkSourceChan::SSBO,
            .format = SinkSourceFormat::Byte,
            .buffer_roi_id = graph.dummy_roi.value(),
        });
      }

      Node node;
      node.sinksources = std::move(sinksources);
      ComputeDispatch node_compute_dispatch;
      node_compute_dispatch.binary_id = compute_dispatch->binary_id();
      node_compute_dispatch.workgroup_count_x = Symbol{
          .type = compute_dispatch->workgroup_count_x_type(),
          .ptr = compute_dispatch->workgroup_count_x(),
      };
      node_compute_dispatch.workgroup_count_y = Symbol{
          .type = compute_dispatch->workgroup_count_y_type(),
          .ptr = compute_dispatch->workgroup_count_y(),
      };
      node_compute_dispatch.workgroup_count_z = Symbol{
          .type = compute_dispatch->workgroup_count_z_type(),
          .ptr = compute_dispatch->workgroup_count_z(),
      };
      uint16_t pc_size = compute_dispatch->push_constant()->size();
      uint32_t pc_count = compute_dispatch->push_constant()->fields()->size();
      std::vector<PushConstantField> fields(pc_count);
      for (uint32_t p = 0; p < pc_count; ++p) {
        const auto *field = compute_dispatch->push_constant()->fields()->Get(p);
        fields[p].offset = field->offset();
        fields[p].value = Symbol{
            .type = field->source_type(),
            .ptr = field->source(),
        };
        switch (field->dtype()) {
        case denox::dnx::ScalarType_I16:
          fields[p].type = PushConstantType::I16;
          break;
        case denox::dnx::ScalarType_U16:
          fields[p].type = PushConstantType::U16;
          break;
        case denox::dnx::ScalarType_I32:
          fields[p].type = PushConstantType::I32;
          break;
        case denox::dnx::ScalarType_U32:
          fields[p].type = PushConstantType::U32;
          break;
        case denox::dnx::ScalarType_I64:
          fields[p].type = PushConstantType::I64;
          break;
        case denox::dnx::ScalarType_U64:
          fields[p].type = PushConstantType::U64;
          break;
        case denox::dnx::ScalarType_F16:
        case denox::dnx::ScalarType_F32:
        case denox::dnx::ScalarType_F64:
          throw std::runtime_error("vkdt_denox does currently not support "
                                   "floating point push constants.");
        }
      }
      node_compute_dispatch.pc = PushConstants{
          .size = pc_size,
          .fields = std::move(fields),
      };
      node.op = std::move(node_compute_dispatch);
      graph.nodes.push_back(std::move(node));

    } else {
      throw std::runtime_error("Unexpected dispatch type.");
    }
  }

  return graph;
}
