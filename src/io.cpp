#include "io.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

static void atomic_write_raw(const std::string &path, const char *data,
                             std::size_t size) {
  const std::filesystem::path fpath(path);
  const std::filesystem::path tmp = fpath.string() + ".tmp";
  {
    std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
    if (!file) {
      throw std::runtime_error("atomic_write: cannot open temp file: " +
                               tmp.string());
    }

    file.write(data, static_cast<std::streamsize>(size));
    if (!file) {
      throw std::runtime_error("atomic_write: write failed for temp file: " +
                               tmp.string());
    }

    file.flush();
    if (!file) {
      throw std::runtime_error("atomic_write: flush failed: " + tmp.string());
    }
  }

  // 2. Atomically replace target
  std::error_code ec;
  std::filesystem::rename(tmp, fpath, ec);
  if (ec) {
    std::filesystem::remove(tmp);
    throw std::runtime_error("atomic_write: rename failed: " + ec.message());
  }
}

void vkdt_denox::write_file_bytes(const std::string &path, const void *buf,
                                  std::size_t size) {
  atomic_write_raw(path, static_cast<const char *>(buf), size);
}

void vkdt_denox::write_file(const std::string &path, const std::string &data) {
  atomic_write_raw(path, data.data(), data.size());
}

std::vector<std::uint8_t> vkdt_denox::read_file_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("read_file_bytes: cannot open " + path);

  f.seekg(0, std::ios::end);
  auto size = f.tellg();
  if (size < 0)
    throw std::runtime_error("read_file_bytes: tellg failed " + path);
  f.seekg(0, std::ios::beg);

  std::vector<std::uint8_t> buf(static_cast<std::size_t>(size));
  if (size != 0) {
    f.read(reinterpret_cast<char *>(buf.data()), size);
    if (!f)
      throw std::runtime_error("read_file_bytes: read failed " + path);
  }
  return buf;
}

std::string vkdt_denox::read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("read_file: cannot open " + path);

  f.seekg(0, std::ios::end);
  auto size = f.tellg();
  if (size < 0)
    throw std::runtime_error("read_file: tellg failed " + path);
  f.seekg(0, std::ios::beg);

  std::string s;
  s.resize(static_cast<std::size_t>(size));

  if (size != 0) {
    f.read(s.data(), size);
    if (!f)
      throw std::runtime_error("read_file: read failed " + path);
  }
  return s;
}

void vkdt_denox::mkdir(const std::filesystem::path &path) {
  if (std::filesystem::exists(path)) {
    if (!std::filesystem::is_directory(path)) {
      throw std::runtime_error("Failed to create output directory. Path "
                               "exists, but is not a directory");
    }
  } else {
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Failed to create output directory");
    }
  }
}
