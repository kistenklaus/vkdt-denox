#include "compute_graph.hpp"
#include "compress_weights.hpp"
#include "symbolics.hpp"
#include <algorithm>
#include <dnx.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

static size_t sizeof_format(vkdt_denox::SinkSourceFormat format) {
  switch (format) {
  case vkdt_denox::SinkSourceFormat::F16:
    return 2;
  case vkdt_denox::SinkSourceFormat::Byte:
    return 1;
  case vkdt_denox::SinkSourceFormat::Auto:
    throw std::runtime_error("trying to get sizeof auto format!");
  }
}

vkdt_denox::ComputeGraph vkdt_denox::reconstruct_compute_graph(
    const denox::dnx::Model *dnx, const CompressedWeights &compressed_weights) {
  const uint32_t buffer_count = dnx->buffers()->size();
  const uint32_t tensor_count = dnx->tensors()->size();
  const uint32_t dispatch_count = dnx->dispatches()->size();
  std::unordered_map<std::string, uint32_t> names;

  // maps buffer ids to owning nodes.
  struct BufferLocation {
    uint32_t owning_node;
    uint32_t borrowing_node;
    uint32_t sinksource_id;
    uint32_t buffer_roi_id;
    uint64_t buffer_ssbo_offset;
  };
  std::vector<BufferLocation> buffer_locations( //
      buffer_count,                             //
      BufferLocation{
          .owning_node = none_sentinal,
          .borrowing_node = none_sentinal,
          .sinksource_id = 0,
          .buffer_roi_id = none_sentinal,
          .buffer_ssbo_offset = 0,
      });

  // Create weight node.
  ComputeGraph graph;
  uint32_t weight_buffer_roi_id = graph.buffer_rois.size();
  graph.buffer_rois.push_back(BufferRoi{
      .byte_size = compressed_weights.data.size(),
      .format = SinkSourceFormat::Byte,
  });
  size_t weight_node_id = graph.nodes.size();
  graph.nodes.push_back(Node{
      .op =
          Upload{
              .name = "weights",
              .sinksource_id = 0,
          },
      .sinksources = {SinkSource{
          .name = "w",
          .type = SinkSourceType::Source,
          .chan = SinkSourceChan::SSBO,
          .format = SinkSourceFormat::Byte,
          .buffer_roi_id = weight_buffer_roi_id,
          .buffer_ssbo_offset = size_t(0),
          .tensor_offset = std::nullopt,
          .tensor_info = nullptr,
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
    assert(compressed_weights.offsets[tensor_id] >= 0);
    buffer_locations[buffer_id].buffer_ssbo_offset =
        static_cast<uint64_t>(compressed_weights.offsets[tensor_id]);
  }

  // Write rois for input!
  for (uint32_t i = 0; i < dnx->inputs()->size(); ++i) {
    const uint32_t tensor_id = dnx->inputs()->Get(i);
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const auto *tensor_info = tensor->info();
    const uint32_t buffer_id = tensor->buffer();
    const auto *buffer = dnx->buffers()->Get(buffer_id);
    const uint32_t input_roi_id = graph.buffer_rois.size();

    SinkSourceFormat format;
    switch (tensor_info->type()) {
    case denox::dnx::ScalarType_I16:
    case denox::dnx::ScalarType_U16:
    case denox::dnx::ScalarType_I32:
    case denox::dnx::ScalarType_U32:
    case denox::dnx::ScalarType_I64:
    case denox::dnx::ScalarType_U64:
    case denox::dnx::ScalarType_F32:
    case denox::dnx::ScalarType_F64:
      throw std::runtime_error("unsupported tensor type!");
    case denox::dnx::ScalarType_F16:
      format = SinkSourceFormat::F16;
      break;
    default:
      throw std::runtime_error("unexpected tensor type!");
    }

    // TODO: Change to extent and type based semantics.
    graph.buffer_rois.push_back(BufferRoi{
        .byte_size =
            Symbol{
                .type = buffer->size_type(),
                .ptr = buffer->size(),
            },
        .format = format,
    });

    buffer_locations[buffer_id].owning_node = external_sential;
    buffer_locations[buffer_id].sinksource_id = i;
    buffer_locations[buffer_id].borrowing_node = none_sentinal;
    buffer_locations[buffer_id].buffer_roi_id = input_roi_id;
    assert(tensor->offset_type() == denox::dnx::ScalarSource_literal);
    assert(read_unsigned_scalar_literal(tensor->offset_as_literal()) == 0);
    buffer_locations[buffer_id].buffer_ssbo_offset = 0;
  }

  for (uint32_t d = 0; d < dispatch_count; ++d) {
    uint32_t node_id = graph.nodes.size();
    const auto *compute_dispatch = dnx->dispatches()->Get(d);

    struct TensorBinding {
      uint16_t set;
      uint16_t binding;
      uint8_t access;
      uint32_t tensor;
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
            .tensor = tensor_id,
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
                    .format = SinkSourceFormat::Byte,
                });
              }
              const uint32_t dummy_roi = graph.dummy_roi.value();
              node_c.dummy_source = node_c.sinksources.size();
              node_c.sinksources.push_back(SinkSource{
                  .name = "dummy",
                  .type = SinkSourceType::Write,
                  .chan = SinkSourceChan::SSBO,
                  .format = SinkSourceFormat::Byte,
                  .buffer_roi_id = dummy_roi,
                  .buffer_ssbo_offset = 0,
                  .tensor_offset = std::nullopt,
                  .tensor_info = nullptr,
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
              .format = SinkSourceFormat::Byte,
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
                  .format = SinkSourceFormat::Byte,
              });
            }
            const uint32_t dummy_roi = graph.dummy_roi.value();

            node_c.dummy_source = node_c.sinksources.size();
            node_c.sinksources.push_back(SinkSource{
                .name = "dummy",
                .type = SinkSourceType::Write,
                .chan = SinkSourceChan::SSBO,
                .format = SinkSourceFormat::Byte,
                .buffer_roi_id = dummy_roi,
                .buffer_ssbo_offset = 0,
                .tensor_offset = std::nullopt,
                .tensor_info = nullptr,
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

      SinkSourceFormat format = SinkSourceFormat::Byte;
      if (type == SinkSourceType::Read) {
        format = SinkSourceFormat::Auto;
      }
      assert(location.owning_node != none_sentinal);
      SinkSourceChan chan = SinkSourceChan::SSBO;
      // TODO: Set appropriate format and roi for output tensors.

      const denox::dnx::TensorInfo *tensor_info = nullptr;
      const denox::dnx::Tensor *tensor = dnx->tensors()->Get(binding.tensor);
      if (tensor != nullptr && tensor->info() != nullptr) {
        tensor_info = tensor->info();
      }

      sinksources.push_back(SinkSource{
          .name = fmt::format("{}", char('a' + b)),
          .type = type,
          .chan = chan,
          .format = format,
          .buffer_roi_id = location.buffer_roi_id,
          .buffer_ssbo_offset = location.buffer_ssbo_offset,
          .tensor_offset = Symbol{tensor->offset_type(), tensor->offset()},
          .tensor_info = tensor_info,
      });
    }

    uint32_t dummy_sink_count = dummy_sink_id - bindings.size();

    if (dummy_sink_count != 0 && !graph.dummy_roi.has_value()) {
      graph.dummy_roi = graph.buffer_rois.size();
      graph.buffer_rois.push_back(BufferRoi{
          .byte_size = 1ull,
          .format = SinkSourceFormat::Byte,
      });
    }

    for (uint32_t i = 0; i < dummy_sink_count; ++i) {

      sinksources.push_back(SinkSource{
          .name = fmt::format("z{}", i),
          .type = SinkSourceType::Read,
          .chan = SinkSourceChan::SSBO,
          .format = SinkSourceFormat::Byte,
          .buffer_roi_id = graph.dummy_roi.value(),
          .buffer_ssbo_offset = 0,
          .tensor_offset = std::nullopt,
          .tensor_info = nullptr,
      });
    }

    Node node;
    node.sinksources = std::move(sinksources);
    ComputeDispatch node_compute_dispatch;
    node_compute_dispatch.info = compute_dispatch->info();
    if (compute_dispatch->info() && compute_dispatch->info()->name()) {
      std::string name = compute_dispatch->info()->name()->str();
      std::replace(name.begin(), name.end(), '-', '_');
      std::replace(name.begin(), name.end(), '+', '_');

      if (names.contains(name)) {
        uint32_t suffix = names[name];
        names[name] += 1;
        name = fmt::format("{}_{}", name, suffix);
      } else {
        names[name] = 1;
      }

      node_compute_dispatch.name = name;
    } else {
      node_compute_dispatch.name = fmt::format("unnamed_dispatch_{}", d);
    }

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
  }

  const uint32_t output_count = dnx->outputs()->size();
  for (uint32_t o = 0; o < output_count; ++o) {
    const uint32_t tensor_id = dnx->outputs()->Get(o);
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const auto *tensor_info = tensor->info();
    const uint32_t buffer_id = tensor->buffer();
    auto &location = buffer_locations[buffer_id];
    assert(location.owning_node != external_sential);
    if (location.owning_node == none_sentinal) {
      throw std::runtime_error("Model does not produce a output, vkdt_denox "
                               "requires at least one output.");
    }
    if (location.borrowing_node != none_sentinal) {
      throw std::runtime_error(
          "vkdt_denox does not support this Model: "
          "Implementation would require a dummy module "
          "connector, which is currently not planned to be implemented!");
    }
    graph.connectors.push_back(Connector{
        .src_node = location.owning_node,
        .src_node_sinksource = location.sinksource_id,
        .dst_node = external_sential,
        .dst_node_sinksource = o,
    });
  }

  // repair input & output rois and add type and meta information.
  const uint32_t input_count = dnx->inputs()->size();
  graph.input_descriptors.resize(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    const uint32_t tensor_id = dnx->inputs()->Get(i);
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const auto *tensor_info = tensor->info();
    const uint32_t buffer_id = tensor->buffer();
    auto &location = buffer_locations[buffer_id];
    assert(location.owning_node == external_sential);
    assert(location.sinksource_id == i);
    auto &roi = graph.buffer_rois[location.buffer_roi_id];
    // TODO: Possibly modify roi

    std::string name;
    if (tensor_info->name() == nullptr) {
      name = fmt::format("unnamed-input-{}", i);
    } else {
      name = tensor_info->name()->str();
    }

    SinkSourceFormat format;
    switch (tensor_info->type()) {
    case denox::dnx::ScalarType_I16:
    case denox::dnx::ScalarType_U16:
    case denox::dnx::ScalarType_I32:
    case denox::dnx::ScalarType_U32:
    case denox::dnx::ScalarType_I64:
    case denox::dnx::ScalarType_U64:
      throw std::runtime_error(
          "vkdt_denox does not support (i8,u8,i16,u16,i32,u32,i64,u64) input / "
          "output types.");
    case denox::dnx::ScalarType_F32:
    case denox::dnx::ScalarType_F64:
      throw std::runtime_error("vkdt_denox does not support (f32, f64) input / "
                               "output types.");
    case denox::dnx::ScalarType_F16:
      format = SinkSourceFormat::F16;
      break;
    default:
      throw std::runtime_error("unexpected scalar type!");
    }

    SinkSourceChan chan = SinkSourceChan::SSBO;
    InOutLayout layout;
    switch (tensor_info->format()) {
    case denox::dnx::TensorFormat_SSBO_HWC:
      layout = InOutLayout::HWC;
      break;
    case denox::dnx::TensorFormat_SSBO_CHW:
      layout = InOutLayout::CHW;
      break;
    case denox::dnx::TensorFormat_SSBO_CHWC8:
      layout = InOutLayout::CHWC8;
      break;
    case denox::dnx::TensorFormat_UNKNOWN:
      throw std::runtime_error("invalid dnx: input with unknown format!");
    case denox::dnx::TensorFormat_TEX_RGBA:
    case denox::dnx::TensorFormat_TEX_RGB:
    case denox::dnx::TensorFormat_TEX_RG:
    case denox::dnx::TensorFormat_TEX_R:
      throw std::runtime_error("texture formats not implemented");
    }
    graph.input_descriptors[i].name = name;
    graph.input_descriptors[i].type = SinkSourceType::Read;
    graph.input_descriptors[i].format = format;
    graph.input_descriptors[i].chan = chan;
    graph.input_descriptors[i].layout = layout;
  }

  graph.output_descriptors.resize(output_count);
  for (uint32_t o = 0; o < output_count; ++o) {
    const uint32_t tensor_id = dnx->outputs()->Get(o);
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const auto *tensor_info = tensor->info();
    const uint32_t buffer_id = tensor->buffer();
    auto &location = buffer_locations[buffer_id];
    assert(location.owning_node != none_sentinal);
    assert(location.owning_node != external_sential);

    auto &roi = graph.buffer_rois[location.buffer_roi_id];
    // TODO: Possibly update buffer roi.

    std::string name;
    if (tensor_info->name() == nullptr) {
      name = fmt::format("unnamed-output-{}", o);
    } else {
      name = tensor_info->name()->str();
    }

    SinkSourceFormat format;
    switch (tensor_info->type()) {
    case denox::dnx::ScalarType_I16:
    case denox::dnx::ScalarType_U16:
    case denox::dnx::ScalarType_I32:
    case denox::dnx::ScalarType_U32:
    case denox::dnx::ScalarType_I64:
    case denox::dnx::ScalarType_U64:
      throw std::runtime_error(
          "vkdt_denox does not support (i8,u8,i16,u16,i32,u32,i64,u64) input / "
          "output types.");
    case denox::dnx::ScalarType_F32:
    case denox::dnx::ScalarType_F64:
      throw std::runtime_error("vkdt_denox does not support (f32, f64) input / "
                               "output types.");
    case denox::dnx::ScalarType_F16:
      format = SinkSourceFormat::F16;
      break;
    default:
      throw std::runtime_error("unexpected scalar type!");
    }

    SinkSourceChan chan = SinkSourceChan::SSBO;
    InOutLayout layout;
    switch (tensor_info->format()) {
    case denox::dnx::TensorFormat_SSBO_HWC:
      layout = InOutLayout::HWC;
      break;
    case denox::dnx::TensorFormat_SSBO_CHW:
      layout = InOutLayout::CHW;
      break;
    case denox::dnx::TensorFormat_SSBO_CHWC8:
      layout = InOutLayout::CHWC8;
      break;
    case denox::dnx::TensorFormat_UNKNOWN:
      throw std::runtime_error("invalid dnx: output with unknown format!");
    case denox::dnx::TensorFormat_TEX_RGBA:
    case denox::dnx::TensorFormat_TEX_RGB:
    case denox::dnx::TensorFormat_TEX_RG:
    case denox::dnx::TensorFormat_TEX_R:
      throw std::runtime_error("texture formats are not implemented.");
    }
    graph.output_descriptors[o].name = name;
    graph.output_descriptors[o].type = SinkSourceType::Write;
    graph.output_descriptors[o].format = format;
    graph.output_descriptors[o].chan = chan;
    graph.output_descriptors[o].layout = layout;

    // update format of owning nodes sinksource
    graph.nodes[location.owning_node]
        .sinksources[location.sinksource_id]
        .format = format;
    graph.buffer_rois[location.buffer_roi_id].format = format;
  }

  // Infer all auto types.
  //    Iterate over all connectors: From construction order we know that
  //    all connectors are topologicalls ordered, we also know that all
  //    edges are RAW connectors.
  //    For each edge where dst-node::dst_sinksource has format auto, infer
  //    type from src-node.
  //    Special-Cases:
  //      If src_node is external, lookup type information in
  //      input_descriptors. If dst_node is external, lookup type information
  //      in output_descriptors.
  for (uint32_t i = 0; i < graph.connectors.size(); ++i) {
    const auto &connector = graph.connectors[i];
    assert(connector.src_node != none_sentinal);
    assert(connector.dst_node != none_sentinal);

    if (connector.src_node == external_sential) {
      assert(connector.dst_node != external_sential);
      const auto &input_description =
          graph.input_descriptors[connector.src_node_sinksource];
      graph.nodes[connector.dst_node]
          .sinksources[connector.dst_node_sinksource]
          .format = input_description.format;
    } else if (connector.dst_node == external_sential) {
      // simply skip
    } else {
      const auto &src = graph.nodes[connector.src_node];
      assert(connector.src_node != external_sential);
      assert(connector.dst_node != external_sential);

      graph.nodes[connector.dst_node]
          .sinksources[connector.dst_node_sinksource]
          .format = graph.nodes[connector.src_node]
                        .sinksources[connector.src_node_sinksource]
                        .format;
    }
  }

  return graph;
}
