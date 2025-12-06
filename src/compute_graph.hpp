#pragma once

#include "symbolics.hpp"
#include <cstdint>
#include <dnx.h>
#include <string>
#include <vector>
namespace vkdt_denox {

enum class SinkSourceType {
  SinkRead,    // <- reads resource
  SourceWrite, // <- writes to resource, this implicitly creates a
               //    resource.
  SourceConst, // <- creates resource
};

enum class Storage {
  SSBO,
};

enum class Dtype {
  U32,
  I32,
  F16,
  Any,
};

struct SinkSource {
  SinkSourceType type;
  std::optional<std::span<const uint8_t>> constSourceData;
  Storage storage;
  Dtype dtype;
};

struct PushConstant {
  uint32_t offset;
  Symbol value;
  Dtype type;
};

struct ComputeDispatch {
  uint32_t spv_binary_id;
  Symbol workgroup_count_x;
  Symbol workgroup_count_y;
  Symbol workgroup_count_z;
  std::vector<PushConstant> push_constants;
};

struct Node {
  std::optional<ComputeDispatch> dispatch;
  std::vector<SinkSource> sinksources;
};

struct Connector {
  int src_node_id;
  std::string src_source_name;
  int dst_node_id;
  std::string dst_sink_name;
};

struct ComputeGraph {
  // Simple adj list
  std::vector<Node> nodes;
  std::vector<Connector> connectors;
};

ComputeGraph reconstruct_compute_graph(const denox::dnx::Model *dnx);

} // namespace vkdt_denox
