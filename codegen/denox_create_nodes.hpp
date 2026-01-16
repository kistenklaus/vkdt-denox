#pragma once

#include "compress_weights.hpp"
#include "compute_graph.hpp"
#include "shader_registry.hpp"
#include "source_writer.hpp"
#include "symbolics.hpp"
#include <dnx.h>
namespace vkdt_denox {

void def_func_denox_create_nodes(SourceWriter &src, const denox::dnx::Model *dnx,
                          const SymbolicIR &symbolic_ir,
                          const ShaderRegistry &shader_registery,
                          const CompressedWeights &compresed_weights,
                          const ComputeGraph &compute_graph,
                          const std::string_view module_name);

} // namespace vkdt_denox
