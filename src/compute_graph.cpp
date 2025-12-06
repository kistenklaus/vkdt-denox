#include "compute_graph.hpp"
#include <dnx.h>
#include <fmt/base.h>
#include <stdexcept>
#include <unordered_map>

vkdt_denox::ComputeGraph
vkdt_denox::reconstruct_compute_graph(const denox::dnx::Model *dnx) {
  ComputeGraph graph;

  // First collect create all nodes and maintain some meta information.
  // 1. Collect all buffer initalizers and create a node for each one.

  uint32_t tensor_count = dnx->tensors()->size();
  std::vector<uint32_t> tensor_id_to_node_map(tensor_count, -1);

  const uint32_t initalizer_count = dnx->initializers()->size();
  for (uint32_t i = 0; i < initalizer_count; ++i) {
    const auto *initalizer = dnx->initializers()->Get(i);

    std::vector<SinkSource> sources = {
        SinkSource{
            .type = SinkSourceType::SourceConst,
            .constSourceData =
                std::span<const uint8_t>{
                    initalizer->data()->data(),
                    initalizer->data()->size(),
                },
            .storage = Storage::SSBO,
            .dtype = Dtype::Any,
        },
    };

    Node node{
        .dispatch = std::nullopt,
        .sinksources = std::move(sources),
    };
    const uint32_t node_id = graph.nodes.size();
    tensor_id_to_node_map[initalizer->tensor()] = node_id;
    graph.nodes.push_back(std::move(node));
  }

  const uint32_t dispatch_count = dnx->dispatches()->size();
  for (uint32_t d = 0; d < dispatch_count; ++d) {
    const auto *dispatch = dnx->dispatches()->Get(d);
    const auto dispatch_type = dnx->dispatches_type()->Get(d);

    if (dispatch_type == denox::dnx::Dispatch_ComputeDispatch) {
      ComputeDispatch vkdt_compute_dispatch;
      std::vector<SinkSource> sinksources;
      const auto *compute_dispatch =
          static_cast<const denox::dnx::ComputeDispatch *>(dispatch);

      vkdt_compute_dispatch.workgroup_count_x = Symbol{
          .type = compute_dispatch->workgroup_count_x_type(),
          .ptr = compute_dispatch->workgroup_count_x(),
      };
      vkdt_compute_dispatch.workgroup_count_y = Symbol{
          .type = compute_dispatch->workgroup_count_y_type(),
          .ptr = compute_dispatch->workgroup_count_y(),
      };
      vkdt_compute_dispatch.workgroup_count_z = Symbol{
          .type = compute_dispatch->workgroup_count_z_type(),
          .ptr = compute_dispatch->workgroup_count_z(),
      };

      uint32_t push_constant_count =
          compute_dispatch->push_constant()->fields()->size();
      for (uint32_t p = 0; p < push_constant_count; ++p) {
        const auto *push_constant =
            compute_dispatch->push_constant()->fields()->Get(p);
        Dtype type;
        switch (push_constant->dtype()) {
        case denox::dnx::ScalarType_I32:
          type = Dtype::U32;
          break;
        case denox::dnx::ScalarType_U32:
          type = Dtype::I32;
          break;
        case denox::dnx::ScalarType_I16:
        case denox::dnx::ScalarType_U16:
        case denox::dnx::ScalarType_I64:
        case denox::dnx::ScalarType_U64:
        case denox::dnx::ScalarType_F16:
        case denox::dnx::ScalarType_F32:
        case denox::dnx::ScalarType_F64:
          throw std::runtime_error("not implemented");
          break;
        }

        vkdt_compute_dispatch.push_constants.push_back(PushConstant{
            .offset = push_constant->offset(),
            .value =
                Symbol{
                    .type = push_constant->source_type(),
                    .ptr = push_constant->source(),
                },
            .type = type,
        });
      }

      vkdt_compute_dispatch.spv_binary_id = compute_dispatch->binary_id();

      const uint32_t descriptor_set_count =
          compute_dispatch->bindings()->size();
      for (uint32_t i = 0; i < descriptor_set_count; ++i) {
        const auto *descriptor_set_binding =
            compute_dispatch->bindings()->Get(i);
        const uint32_t descriptor_count =
            descriptor_set_binding->bindings()->size();
        for (uint32_t j = 0; j < descriptor_count; ++j) {
          const auto *descriptor_binding =
              descriptor_set_binding->bindings()->Get(j);

          const uint32_t tensor = descriptor_binding->tensor();
          const uint32_t binding = descriptor_binding->binding();
          const denox::dnx::Access access = descriptor_binding->access();
          switch (access) {
          case denox::dnx::Access_ReadOnly: {
            // make into a sinkk
            sinksources.push_back(SinkSource{
                .type = SinkSourceType::SinkRead,
                .constSourceData = std::nullopt,
                .storage = Storage::SSBO,
                .dtype = Dtype::Any,
            });
            break;
          }
          case denox::dnx::Access_WriteOnly: {
            // make into a source.
            sinksources.push_back(SinkSource{
                .type = SinkSourceType::SourceWrite,
                .constSourceData = std::nullopt,
                .storage = Storage::SSBO,
                .dtype = Dtype::Any,
            });
            // write only resources are always implicitly created at this point. <- NOT CORRECT!!!

            
            break;
          }
          case denox::dnx::Access_ReadWrite:
            throw std::runtime_error(
                "ReadWrite tensor bindings are not supported, for vkdt_denox!");
            break;
          }
        }
      }

      Node node{
          .dispatch = std::move(vkdt_compute_dispatch),
          .sinksources = std::move(sinksources),
      };
      graph.nodes.push_back(std::move(node));

    } else {
      throw std::runtime_error("not implemented");
    }
  }

  fmt::println("{:=^40}", "Nodes");
  for (uint32_t n = 0; n < graph.nodes.size(); ++n) {
    const auto &node = graph.nodes[n];
    if (node.dispatch.has_value()) {
      fmt::println("node-{} (compute-dispatch)", n);
    } else {
      fmt::println("node-{} (weight)", n);
    }
    for (uint32_t i = 0; i < node.sinksources.size(); ++i) {
      const auto &sinksource = node.sinksources[i];
      fmt::print(" - (b{}, ", i);

      switch (sinksource.type) {
      case SinkSourceType::SinkRead:
        fmt::print("read");
        break;
      case SinkSourceType::SourceWrite:
        fmt::print("write");
        break;
      case SinkSourceType::SourceConst:
        fmt::print("source");
        break;
      }
      fmt::print(", ");
      switch (sinksource.storage) {
      case Storage::SSBO:
        fmt::print("ssbo");
        break;
      }
      fmt::print(", ");
      switch (sinksource.dtype) {
      case Dtype::F16:
        fmt::print("f16");
        break;
      case Dtype::Any:
        fmt::print("*");
        break;
      case Dtype::U32:
        fmt::print("u32");
        break;
      case Dtype::I32:
        fmt::print("i32");
        break;
      }
      fmt::println(")");
    }
  }

  return graph;
}
