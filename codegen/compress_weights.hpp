#pragma once

#include <cstddef>
#include <dnx.h>
namespace vkdt_denox {

struct CompressedWeights {
  // Maps tensor ids to compressed weight offsets.
  // if offsets[tensor-id] == -1, then this tensor is not a weight!
  // Otherwise gives aligned offset of the tensor-id.
  std::vector<int64_t> offsets;
  std::vector<uint8_t> data;
};

CompressedWeights compress_weights(const denox::dnx::Model *model);

} // namespace vkdt_denox
