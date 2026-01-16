#include "symbolics.hpp"
#include <dnx.h>
#include <fmt/format.h>
#include <stdexcept>

vkdt_denox::SymbolicIR
vkdt_denox::read_symbolic_ir(const denox::dnx::Model *dnx) {
  SymbolicIR ir;
  const uint32_t var_count = dnx->sym_ir()->var_count();
  const uint32_t symbol_count = dnx->sym_ir()->ops()->size() + var_count;
  ir.symir = dnx->sym_ir();

  const auto *valueNames = dnx->value_names();

  ir.vars.resize(var_count);
  std::vector<bool> set(var_count, false);

  for (uint32_t i = 0; i < valueNames->size(); ++i) {
    const auto *valueName = valueNames->Get(i);
    if (valueName->value_type() == denox::dnx::ScalarSource_literal) {
      continue; // we don't care about constant names here
    }
    assert(valueName->value_type() == denox::dnx::ScalarSource_symbolic);
    const auto *symref = valueName->value_as_symbolic();
    uint32_t sid = symref->sid();
    if (sid >= var_count) {
      // we don't care about intermediate value names!
      continue;
    }
    std::string name = valueName->name()->str();
    ir.vars[sid] = name;
    set[sid] = true;
  }
  for (uint32_t i = 0; i < var_count; ++i) {
    if (set[i] == false) {
      throw std::runtime_error(
          "vkdt-denox requires all dynamic extents to have names!");
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
