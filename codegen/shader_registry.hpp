#pragma once

#include <cstdint>
#include <dnx.h>
#include <span>
#include <string>
#include <vector>
namespace vkdt_denox {

struct ShaderBinary {
  std::string name;
  std::span<const uint32_t> spv;
};

struct ShaderRegistry {
  std::vector<ShaderBinary> binaries;
};

ShaderRegistry create_shader_registry(const denox::dnx::Model *dnx);

} // namespace vkdt_denox
