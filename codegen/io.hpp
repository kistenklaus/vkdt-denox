#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
namespace vkdt_denox {

void write_file_bytes(const std::string &path, const void *buf, std::size_t size);

void write_file(const std::string &path, const std::string &data);

std::vector<std::uint8_t> read_file_bytes(const std::string &path);

std::string read_file(const std::string &path);

void mkdir(const std::filesystem::path &path);

} // namespace vkdt_denox
