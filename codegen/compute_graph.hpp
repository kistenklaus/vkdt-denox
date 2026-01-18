#pragma once

#include "compress_weights.hpp"
#include "symbolics.hpp"
#include <cstdint>
#include <dnx.h>
#include <string>
#include <variant>
#include <vector>
namespace vkdt_denox {

static constexpr uint32_t none_sentinal = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t external_sential =
    std::numeric_limits<uint32_t>::max() - 1;

enum class SinkSourceType {
  Read,
  Write,
  Source,
};

enum class SinkSourceChan {
  SSBO,
};

enum class SinkSourceFormat {
  F16,
  Byte,
  Auto,
};

struct SinkSource {
  std::string name;
  SinkSourceType type;
  SinkSourceChan chan;
  SinkSourceFormat format;
  uint32_t buffer_roi_id;
  std::variant<Symbol, uint64_t> ssbo_offset;
  const denox::dnx::TensorInfo* tensor_info;
};

enum PushConstantType {
  U32,
  I32,
  U16,
  I16,
  U64,
  I64,
};

struct PushConstantField {
  uint16_t offset;
  PushConstantType type;
  Symbol value;
};

struct PushConstants {
  uint16_t size;
  std::vector<PushConstantField> fields;
};

struct ComputeDispatch {
  std::string name;
  uint32_t binary_id;
  Symbol workgroup_count_x;
  Symbol workgroup_count_y;
  Symbol workgroup_count_z;
  PushConstants pc;
  const denox::dnx::DispatchInfo* info;
};

struct Upload {
  std::string name;
  uint32_t sinksource_id;
};

struct Node {
  std::variant<ComputeDispatch, Upload> op;
  std::vector<SinkSource> sinksources;
  std::optional<uint32_t> dummy_source;
};

struct Connector {
  uint32_t src_node;
  uint32_t src_node_sinksource;
  uint32_t dst_node;
  uint32_t dst_node_sinksource;
};

struct BufferRoi {
  std::variant<Symbol, uint64_t> byte_size;
  std::optional<std::pair<Symbol, Symbol>>
      extent; // (width, height) <- in pixel coordinates!
  SinkSourceFormat format;
};

enum class InOutLayout {
  HWC,
  CHW,
  CHWC8,
};

struct InOutDescriptor {
  std::string name;
  SinkSourceType type;
  SinkSourceChan chan;
  SinkSourceFormat format;
  InOutLayout layout;
};

struct ComputeGraph {
  // Simple adj list
  std::vector<Node> nodes;
  std::vector<Connector> connectors;
  std::vector<BufferRoi> buffer_rois;
  std::optional<uint32_t> dummy_roi;

  std::vector<InOutDescriptor> input_descriptors;
  std::vector<InOutDescriptor> output_descriptors;
};

ComputeGraph
reconstruct_compute_graph(const denox::dnx::Model *dnx,
                          const CompressedWeights &compressed_weights);

} // namespace vkdt_denox
