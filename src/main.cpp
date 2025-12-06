
#include "compute_graph.hpp"
#include "create_nodes.hpp"
#include "io.hpp"
#include "source_writer.hpp"
#include "symbolics.hpp"
#include "tensor_extent.hpp"
#include <dnx.h>
#include <fmt/base.h>
#include <fmt/format.h>

int main() {

  // Read dnx file from disk.
  std::string dnx_path = "./net.dnx";
  std::vector<uint8_t> dnx_buffer = vkdt_denox::read_file_bytes(dnx_path);
  const auto *dnx = denox::dnx::GetModel(dnx_buffer.data());

  // Preprocessing
  vkdt_denox::SymbolicIR ir = vkdt_denox::read_symbolic_ir(dnx);
  vkdt_denox::ComputeGraph compute_graph = vkdt_denox::reconstruct_compute_graph(dnx);


  // Code generation.
  vkdt_denox::SourceWriter src;
  vkdt_denox::def_struct_tensor_extent(src);
  vkdt_denox::def_struct_symbolic_table(src, ir);
  vkdt_denox::def_func_symbolic_expressions(src, ir);

  vkdt_denox::def_func_create_nodes(src, dnx);

  // Write generated code to disk.
  vkdt_denox::write_file("main.c", src.finish());
  
  // fmt::println("{}", src.finish());
}
