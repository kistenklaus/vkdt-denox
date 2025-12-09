
#include "compress_weights.hpp"
#include "compute_graph.hpp"
#include "create_nodes.hpp"
#include "io.hpp"
#include "shader_registry.hpp"
#include "source_writer.hpp"
#include "symbolics.hpp"
#include <dnx.h>
#include <filesystem>
#include <fmt/format.h>

int main() {

  // Read dnx file from disk.
  std::string dnx_path = "./trivial-net.dnx";
  std::vector<uint8_t> dnx_buffer = vkdt_denox::read_file_bytes(dnx_path);
  const auto *dnx = denox::dnx::GetModel(dnx_buffer.data());

  // Preprocessing
  vkdt_denox::SymbolicIR symbolic_ir = vkdt_denox::read_symbolic_ir(dnx);
  vkdt_denox::CompressedWeights compressed_weights =
      vkdt_denox::compress_weights(dnx);
  vkdt_denox::ShaderRegistry shader_registry =
      vkdt_denox::create_shader_registry(dnx);
  vkdt_denox::ComputeGraph compute_graph =
      vkdt_denox::reconstruct_compute_graph(dnx, compressed_weights);

  // Code generation.
  vkdt_denox::SourceWriter src;
  vkdt_denox::def_func_create_nodes(src, dnx, symbolic_ir, shader_registry,
                                    compressed_weights, compute_graph, "denox");

  // Write generated code to disk.
  const std::filesystem::path module_dir = "output/";
  vkdt_denox::mkdir(module_dir);

  // 1. Write compressed weights to disk.
  vkdt_denox::write_file_bytes(module_dir / "weights.bin",
                               compressed_weights.data.data(),
                               compressed_weights.data.size());
  for (const auto &binary : shader_registry.binaries) {
    vkdt_denox::write_file_bytes(module_dir / (binary.name + ".comp.spv"),
                                 binary.spv.data(),
                                 binary.spv.size() * sizeof(uint32_t));
  }
  vkdt_denox::write_file(module_dir / "main.c", src.finish());

  // fmt::println("{}", src.finish());
}
