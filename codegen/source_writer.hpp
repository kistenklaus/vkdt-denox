#pragma once

#include <cassert>
#include <fmt/format.h>
#include <map>
#include <sstream>
#include <string>
#include <utility>
namespace vkdt_denox {

static constexpr uint32_t SPACES_PER_INDENTATION = 2;

enum class IncludeType { System, Local };

class SourceWriter {
private:
  struct Include {
    std::string str;
    IncludeType type;
  };

public:
  void append(const std::string &src) {
    if (src.empty()) {
      return;
    }
    std::stringstream ss(src);
    std::string line;
    while (std::getline(ss, line)) {
      m_code.append(fmt::format("{}{}\n", m_indentation, line));
    }
  }

  void add_include(const std::string &include, IncludeType include_type) {
    if (m_includes.contains(include)) {
      return;
    }
    m_includes.insert(std::make_pair(include, Include{include, include_type}));
  }

  void add_header_guard(const std::string &guard_macro) {
    m_header_guard_macro = guard_macro;
  }

  void push_indentation(uint32_t count = 1) {
    for (uint32_t i = 0; i < count * SPACES_PER_INDENTATION; ++i) {
      m_indentation.push_back(' ');
    }
  }

  void pop_indentation(uint32_t count = 1) {
    std::size_t size = m_indentation.size();
    assert(size >= count * SPACES_PER_INDENTATION); // indentation passed 0.
    std::size_t new_size = size - count * SPACES_PER_INDENTATION;
    m_indentation.resize(new_size);
  }

  std::string finish() {
    std::string preamble;
    std::string epilog;
    if (!m_header_guard_macro.empty()) {
      preamble += fmt::format("#ifndef {}\n", m_header_guard_macro);
      preamble += fmt::format("#define {}\n", m_header_guard_macro);
    }
    for (const auto& [_, inc] : m_includes) {
      if (inc.type == IncludeType::Local) {
        preamble += fmt::format("#include \"{}\"\n", inc.str);
      }
    }
    for (const auto& [_, inc] : m_includes) {
      if (inc.type == IncludeType::System) {
        preamble += fmt::format("#include <{}>\n", inc.str);
      }
    }

    if (!m_header_guard_macro.empty()) {
      epilog += fmt::format("#endif\n");
    }

    std::string src = fmt::format("{}{}{}", preamble, m_code, epilog);
    return src;
  }

private:
  std::string m_header_guard_macro;
  std::map<std::string, Include> m_includes;
  std::string m_indentation;
  std::string m_code;
};

} // namespace vkdt_denox
