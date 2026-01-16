#pragma once

#include "dnx.h"

namespace vkdt_denox {

struct SymbolicIR {
  const denox::dnx::SymIR *symir;
  std::vector<std::string> vars;
};

struct Symbol {
  denox::dnx::ScalarSource type;
  const void *ptr;
};

/// Parses all information related to symbolics out of the dnx file.
SymbolicIR read_symbolic_ir(const denox::dnx::Model *dnx);

uint64_t read_unsigned_scalar_literal(const denox::dnx::ScalarLiteral *literal);

} // namespace vkdt_denox
