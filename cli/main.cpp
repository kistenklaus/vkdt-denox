#include "compress_weights.hpp"
#include "compute_graph.hpp"
#include "denox_create_nodes.hpp"
#include "denox_read_source.hpp"
#include "io.hpp"
#include "shader_registry.hpp"
#include "source_writer.hpp"
#include "symbolics.hpp"
#include <CLI/CLI.hpp>
#include <dnx.h>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  CLI::App app{
      "vkdt-denox — C code generator for vkdt from compiled CNN artifacts"};

  std::string dnx_path_str;
  std::string src_dir_str;
  std::string shader_dir_str;
  std::string weight_dir_str;
  std::string bin_dir_str;
  std::string module_name;
  bool mkdir = false;

  // Positional: DNX artifact
  app.add_option("dnx", dnx_path_str, "Compiled neural network artifact (.dnx)")
      ->required();

  // Required output directories
  app.add_option("--src-dir", src_dir_str,
                 "Output directory for generated C source files")
      ->required();

  app.add_option("--shader-dir", shader_dir_str,
                 "Output directory for generated shader sources")
      ->required();

  app.add_option("--weight-dir", weight_dir_str,
                 "Output directory for neural network weights")
      ->required();

  app.add_option("--bin-dir", bin_dir_str, "vkdt binary directory");

  app.add_option("--module-name", module_name, "Name of the vkdt module")
      ->required();

  // Directory creation flag
  app.add_flag(
      "-p,--mkdir", mkdir,
      "Create output directories (including parents) if they do not exist");

  CLI11_PARSE(app, argc, argv);

  // ---- Filesystem validation ----

  const fs::path dnx_path{dnx_path_str};
  const fs::path src_dir{src_dir_str};
  const fs::path shader_dir{shader_dir_str};
  const fs::path weight_dir{weight_dir_str};
  const fs::path bin_dir{bin_dir_str};

  if (!fs::exists(dnx_path) || !fs::is_regular_file(dnx_path)) {
    std::cerr << "Error: DNX artifact does not exist or is not a regular file: "
              << dnx_path << '\n';
    return 1;
  }

  auto check_output_dir = [&](const fs::path &dir, const char *name) -> bool {
    if (fs::exists(dir)) {
      if (!fs::is_directory(dir)) {
        std::cerr << "Error: " << name
                  << " exists but is not a directory: " << dir << '\n';
        return false;
      }
      return true;
    }

    if (!mkdir) {
      std::cerr << "Error: " << name << " does not exist: " << dir << '\n'
                << "       Use --mkdir (-p) to create it.\n";
      return false;
    }

    std::error_code ec;
    if (!fs::create_directories(dir, ec)) {
      std::cerr << "Error: failed to create directory " << dir << ": "
                << ec.message() << '\n';
      return false;
    }

    return true;
  };

  if (!check_output_dir(src_dir_str, "src-dir") ||
      !check_output_dir(shader_dir_str, "shader-dir") ||
      !check_output_dir(weight_dir_str, "weight-dir") ||
      !check_output_dir(bin_dir_str, "bin-dir")) {
    return 1;
  }

  // Load dnx
  std::vector<uint8_t> dnx_buffer = vkdt_denox::read_file_bytes(dnx_path_str);
  const auto *dnx = denox::dnx::GetModel(dnx_buffer.data());

  // Preprocessing for codegeneration
  vkdt_denox::SymbolicIR symbolic_ir = vkdt_denox::read_symbolic_ir(dnx);

  vkdt_denox::CompressedWeights compressed_weights =
      vkdt_denox::compress_weights(dnx);
  vkdt_denox::ShaderRegistry shader_registry =
      vkdt_denox::create_shader_registry(dnx);
  vkdt_denox::ComputeGraph compute_graph =
      vkdt_denox::reconstruct_compute_graph(dnx, compressed_weights);

  fs::path weight_path =
      weight_dir / fmt::format("{}-weights.dat", module_name);
  std::string weight_path_str = weight_path.string();
  std::string rel_weight_path_str = fs::relative(weight_path, bin_dir).string();
  fmt::println("relative-path: {}", rel_weight_path_str);

  vkdt_denox::write_file_bytes(weight_path_str, compressed_weights.data.data(),
                               compressed_weights.data.size());

  for (const auto &binary : shader_registry.binaries) {
    fs::path path = shader_dir / (binary.name + ".comp.spv");
    vkdt_denox::write_file_bytes(path, binary.spv.data(),
                                 binary.spv.size() * sizeof(uint32_t));
  }

  vkdt_denox::SourceWriter src;
  src.add_header_guard(fmt::format("{}_DENOX_MODULE_H", module_name));
  src.append("\n");
  vkdt_denox::def_func_denox_read_source(src, compute_graph, compressed_weights,
                                         rel_weight_path_str, module_name);

  src.append("\n");
  vkdt_denox::def_func_denox_create_nodes(src, dnx, symbolic_ir,
                                          shader_registry, compressed_weights,
                                          compute_graph, module_name);
  src.append("\n");

  fs::path src_path = src_dir / "denox_model.h";
  vkdt_denox::write_file(src_path, src.finish());
  return 0;
}
