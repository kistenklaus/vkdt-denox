#include "shader_registry.hpp"
#include <fmt/format.h>

vkdt_denox::ShaderRegistry
vkdt_denox::create_shader_registry(const denox::dnx::Model *dnx) {

  const uint32_t binary_count = dnx->shader_binaries()->size();

  ShaderRegistry registry;
  registry.binaries.resize(binary_count);

  for (uint32_t i = 0; i < binary_count; ++i) {
    const auto *binary = dnx->shader_binaries()->Get(i);
    registry.binaries[i].name = fmt::format("comp{}", i);
    registry.binaries[i].spv = std::span<const uint32_t>{
        binary->spirv()->data(), binary->spirv()->size()};
  }

  return registry;
}
