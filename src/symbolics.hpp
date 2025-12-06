#pragma once

#include "dnx.h"
#include "source_writer.hpp"

namespace vkdt_denox {

enum class SymbolicVarSourceDim {
  Height,
  Width,
};

struct SymbolicVarSource {
  uint32_t input_index;
  SymbolicVarSourceDim dim;
};

struct SymbolicIR {
  std::vector<bool> symbol_is_referrenced;
  const denox::dnx::SymIR *symir;
  std::vector<SymbolicVarSource> sym_sources;
};

struct Symbol {
  denox::dnx::ScalarSource type;
  const void *ptr;
};

/// Parses all information related to symbolics out of the dnx file.
SymbolicIR read_symbolic_ir(const denox::dnx::Model *dnx);

/// Defines a struct which holds the result of a symbolic_eval.
/// Returns the name of this struct.
void def_struct_symbolic_table(SourceWriter &src, SymbolicIR &ir);

void def_func_symbolic_expressions(SourceWriter &src, SymbolicIR &ir);

std::string access_symbol(SymbolicIR &ir, std::string_view table_ptr_var,
                          Symbol symbol);

} // namespace vkdt_denox
