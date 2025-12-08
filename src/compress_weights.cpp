#include "compress_weights.hpp"
#include "symbolics.hpp"
#include "util.hpp"
#include <algorithm>
#include <dnx.h>
#include <fmt/base.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

vkdt_denox::CompressedWeights
vkdt_denox::compress_weights(const denox::dnx::Model *dnx) {
  const auto *initalizers = dnx->initializers();

  size_t offset = 0;
  const uint32_t initalizer_count = initalizers->size();
  for (uint32_t i = 0; i < initalizer_count; ++i) {
    const auto *initalizer = initalizers->Get(i);
    const uint32_t tensor_id = initalizer->tensor();
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const uint32_t buffer_id = tensor->buffer();
    const auto *buffer = dnx->buffers()->Get(buffer_id);

    if (tensor->offset_type() == denox::dnx::ScalarSource_symbolic) {
      throw std::runtime_error(
          "Unexpected tensor offset. vkdt_denox assumes that tensor "
          "intializers reference tensors with a compiletime offset, "
          "encountered symbolic expression. Operation not supported!");
    }
    uint64_t tensor_offset =
        vkdt_denox::read_unsigned_scalar_literal(tensor->offset_as_literal());
    if (tensor_offset != 0) {
      throw std::runtime_error(
          "Unexpected tensor offset. vkdt_denox assumes that TensorInitalizers "
          "initalize full buffers, encountered partial initalization of a "
          "buffer, which is currently not implemented!");
    }

    const size_t alignment = buffer->alignment();

    offset = align_up(offset, alignment);
    offset += initalizer->data()->size();
  }
  const size_t byte_size = offset;
  CompressedWeights compressed_weights;
  compressed_weights.offsets.resize(dnx->tensors()->size(), -1);
  compressed_weights.data.resize(byte_size, 0); // <- initalize all to zero!

  offset = 0;
  for (uint32_t i = 0; i < initalizer_count; ++i) {
    const auto *initalizer = initalizers->Get(i);
    const uint32_t tensor_id = initalizer->tensor();
    const auto *tensor = dnx->tensors()->Get(tensor_id);
    const uint32_t buffer_id = tensor->buffer();
    const auto *buffer = dnx->buffers()->Get(buffer_id);
    const size_t alignment = buffer->alignment();

    offset = align_up(offset, alignment);
    std::memcpy(compressed_weights.data.data() + offset,
                initalizer->data()->data(), initalizer->data()->size());
    compressed_weights.offsets[tensor_id] = offset;
    offset += initalizer->data()->size();
  }
  return compressed_weights;
}
