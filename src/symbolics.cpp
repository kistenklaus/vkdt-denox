#include "symbolics.hpp"
#include <dnx.h>
#include <stdexcept>

vkdt_denox::SymbolicIR
vkdt_denox::read_symbolic_ir(const denox::dnx::Model *dnx) {
  SymbolicIR ir;
  const uint32_t var_count = dnx->sym_ir()->var_count();
  const uint32_t symbol_count = dnx->sym_ir()->ops()->size() + var_count;

  // 1. Search for all symbols that are even used anywhere.

  ir.symir = dnx->sym_ir();

  const uint32_t input_count = dnx->inputs()->size();
  ir.sym_sources.resize(var_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    const auto *tensor_info = dnx->inputs()->Get(i);
    if (tensor_info->width_type() == denox::dnx::ScalarSource_symbolic) {
      uint32_t sid = tensor_info->width_as_symbolic()->sid();
      if (sid < var_count) {
        ir.sym_sources[sid].dim = SymbolicVarSourceDim::Width;
        ir.sym_sources[sid].input_index = i;
      }
    }
    if (tensor_info->height_type() == denox::dnx::ScalarSource_symbolic) {
      uint32_t sid = tensor_info->height_as_symbolic()->sid();
      if (sid < var_count) {
        ir.sym_sources[sid].dim = SymbolicVarSourceDim::Height;
        ir.sym_sources[sid].input_index = i;
      }
    }
  }

  return ir;
}

uint64_t vkdt_denox::read_unsigned_scalar_literal(
    const denox::dnx::ScalarLiteral *literal) {
  uint64_t v;
  switch (literal->dtype()) {
  case denox::dnx::ScalarType_I16: {
    int16_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int16_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U16: {
    uint16_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint16_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_I32: {
    int32_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int32_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U32: {
    uint32_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint32_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_I64: {
    int64_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int64_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U64: {
    uint64_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint64_t));
    v = tmp;
    break;
  }
  case denox::dnx::ScalarType_F16:
  case denox::dnx::ScalarType_F32:
  case denox::dnx::ScalarType_F64:
    throw std::runtime_error("Floating point types, are not allowed in "
                             "scalar sources. Invalid dnx file format!");
    break;
  }
  return v;
}
