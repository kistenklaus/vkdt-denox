#pragma once

#include "compress_weights.hpp"
#include "compute_graph.hpp"
#include "source_writer.hpp"
namespace vkdt_denox {

void def_func_denox_read_source(SourceWriter &src,
                                const ComputeGraph &compute_graph,
                                const CompressedWeights &compressed_weights,
                                std::string_view weights_path,
                                std::string_view module_name);

} // namespace vkdt_denox
