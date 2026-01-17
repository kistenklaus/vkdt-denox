#include "denox_create_nodes.hpp"
#include "compute_graph.hpp"
#include "symbolics.hpp"
#include <dnx.h>
#include <fmt/base.h>
#include <stdexcept>
#include <variant>

namespace vkdt_denox {

static void eval_symbolics(SourceWriter &src, const SymbolicIR &ir,
                           const std::vector<bool> &referenced_symbols) {

  const uint32_t m = ir.symir->ops()->size();
  const uint32_t k = ir.vars.size();
  const uint32_t n = k + m;

  std::vector<std::string> symbol_names(n);
  for (size_t i = 0; i < k; ++i) {
    symbol_names[i] = ir.vars[i];
  }

  std::vector<uint32_t> ref_counts(n);
  for (uint32_t i = 0; i < n; ++i) {
    ref_counts[i] = referenced_symbols[i] ? 1 : 0;
  }

  std::vector<std::string> expressions(m);
  for (uint32_t i = 0; i < m; ++i) {
    uint32_t sid = k + i;
    const auto *op = ir.symir->ops()->Get(i);
    const auto opcode = op->opcode();

    std::string lhs;
    if (opcode & denox::dnx::SymIROpCode_LHSC) {
      lhs = fmt::format("{}", op->lhs());
    } else {
      uint32_t sid = op->lhs();
      lhs = symbol_names[sid];
      ++ref_counts[op->lhs()];
    }

    std::string rhs;
    if (opcode & denox::dnx::SymIROpCode_RHSC) {
      rhs = fmt::format("{}", op->rhs());
    } else {
      uint32_t sid = op->rhs();
      rhs = symbol_names[sid];
      ++ref_counts[op->rhs()];
    }

    auto mask = ~denox::dnx::SymIROpCode_LHSC & ~denox::dnx::SymIROpCode_RHSC;

    auto operation = opcode & mask;

    std::string expr;
    if (operation == denox::dnx::SymIROpCode_ADD) {
      expr = fmt::format("{} + {}", lhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_SUB) {
      expr = fmt::format("{} - {}", lhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_MUL) {
      expr = fmt::format("{} * {}", lhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_DIV) {
      expr = fmt::format("{} / {}", lhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_MOD) {
      expr = fmt::format("(({} % {}) + {}) % {}", lhs, rhs, rhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_MIN) {
      expr = fmt::format("{} < {} ? {} : {}", lhs, rhs, lhs, rhs);
    } else if (operation == denox::dnx::SymIROpCode_MAX) {
      expr = fmt::format("{} < {} ? {} : {}", lhs, rhs, rhs, lhs);
    }

    std::string symbol_name = fmt::format("s{}", sid);
    symbol_names[sid] = symbol_name;
    expressions[i] = fmt::format("int64_t {} = {};", symbol_name, expr);
  }

  std::vector<bool> pruned_expressions(m, false);

  while (true) {
    bool pruned_once = false;

    for (uint32_t i = 0; i < m; ++i) {
      uint32_t sid = i + k;
      if (pruned_expressions[i]) {
        continue;
      }
      if (ref_counts[sid] > 0) {
        continue;
      }
      pruned_expressions[i] = true;
      pruned_once = true;
      const auto *op = ir.symir->ops()->Get(i);
      if (!(op->opcode() & denox::dnx::SymIROpCode_LHSC)) {
        uint32_t sid = op->lhs();
        assert(ref_counts[sid] > 0);
        --ref_counts[sid];
      }

      if (!(op->opcode() & denox::dnx::SymIROpCode_RHSC)) {
        uint32_t sid = op->rhs();
        assert(ref_counts[sid] > 0);
        --ref_counts[sid];
      }
    }

    if (!pruned_once) {
      break;
    }
  }

  for (uint32_t i = 0; i < m; ++i) {
    if (ref_counts[k + i] > 0) {
      src.append(expressions[i]);
    }
  }
}

static size_t sinksource_format_size(vkdt_denox::SinkSourceFormat format) {
  switch (format) {
  case vkdt_denox::SinkSourceFormat::F16:
    return 2;
  case vkdt_denox::SinkSourceFormat::Byte:
    return 1;
  case vkdt_denox::SinkSourceFormat::Auto:
    throw std::runtime_error("trying to get sizeof auto type.");
    break;
  }
  throw std::runtime_error("unreachable");
}

static std::string access_symbol(const SymbolicIR &ir,
                                 vkdt_denox::Symbol symbol,
                                 std::vector<bool> &referenced_symbols) {
  if (symbol.type == denox::dnx::ScalarSource_symbolic) {
    const auto *sym_ref = static_cast<const denox::dnx::SymRef *>(symbol.ptr);
    const uint32_t sid = sym_ref->sid();
    referenced_symbols[sid] = true;
    if (sid < ir.vars.size()) {
      return ir.vars[sid];
    } else {
      return fmt::format("s{}", sym_ref->sid());
    }
  } else if (symbol.type == denox::dnx::ScalarSource_literal) {
    return fmt::format(
        "{}", vkdt_denox::read_unsigned_scalar_literal(
                  static_cast<const denox::dnx::ScalarLiteral *>(symbol.ptr)));
  } else {
    throw std::runtime_error("invalid scalar source type!");
  }
}

static void create_buffer_rois(SourceWriter &src, const SymbolicIR &symbolic_ir,
                               const ComputeGraph &compute_graph,
                               std::vector<bool> &referenced_symbols) {
  uint32_t n = compute_graph.buffer_rois.size();
  for (uint32_t i = 0; i < n; ++i) {
    const auto &buffer_roi = compute_graph.buffer_rois[i];
    if (buffer_roi.extent.has_value()) {
      throw std::runtime_error(
          "buffer-rois with extent are not implemented in the codegen!");
    }
    if (std::holds_alternative<size_t>(buffer_roi.byte_size)) {
      size_t byte_size = std::get<size_t>(buffer_roi.byte_size);
      assert(buffer_roi.format != SinkSourceFormat::Auto);
      if (buffer_roi.format != SinkSourceFormat::Byte) {
        uint32_t format_size = sinksource_format_size(buffer_roi.format);
        assert(byte_size % format_size == 0);
        byte_size /= format_size;
      }
      src.append(
          fmt::format("dt_roi_t roi{} = {{.wd = {}, .ht = 1}};", i, byte_size));
    } else if (std::holds_alternative<Symbol>(buffer_roi.byte_size)) {
      const auto &symbol = std::get<Symbol>(buffer_roi.byte_size);

      if (buffer_roi.format != SinkSourceFormat::Byte) {
        uint32_t format_size = sinksource_format_size(buffer_roi.format);
        src.append(fmt::format(
            "dt_roi_t roi{} = {{.wd = (uint32_t)({} / {}), .ht = 1}};", i,
            access_symbol(symbolic_ir, symbol, referenced_symbols),
            format_size));
      } else {
        src.append(fmt::format(
            "dt_roi_t roi{} = {{.wd = (uint32_t)({}), .ht = 1}};", i,
            access_symbol(symbolic_ir, symbol, referenced_symbols)));
      }
    }
  }
}
static std::string_view
push_constant_type_to_string(vkdt_denox::PushConstantType type) {
  switch (type) {
  case vkdt_denox::U32:
    return "uint32_t";
  case vkdt_denox::I32:
    return "int32_t";
  case vkdt_denox::U16:
    return "uint16_t";
  case vkdt_denox::I16:
    return "int16_t";
  case vkdt_denox::U64:
    return "uint64_t";
  case vkdt_denox::I64:
    return "int64_t";
    break;
  }
  throw std::runtime_error("unreachable");
}
static std::string_view
sinksource_type_to_string(vkdt_denox::SinkSourceType sinksource_type) {
  switch (sinksource_type) {
  case vkdt_denox::SinkSourceType::Read:
    return "read";
  case vkdt_denox::SinkSourceType::Write:
    return "write";
  case vkdt_denox::SinkSourceType::Source:
    return "source";
  }
  throw std::runtime_error("unreachable");
}

static std::string_view
sinksource_chan_to_string(vkdt_denox::SinkSourceChan chan) {
  switch (chan) {
  case vkdt_denox::SinkSourceChan::SSBO:
    return "ssbo";
  }
  throw std::runtime_error("unreachable");
}

static std::string_view
sinksource_format_to_string(vkdt_denox::SinkSourceFormat format) {
  switch (format) {
  case vkdt_denox::SinkSourceFormat::F16:
    return "f16";
  case vkdt_denox::SinkSourceFormat::Byte:
    return "u8";
  case vkdt_denox::SinkSourceFormat::Auto:
    return "*";
    break;
  }
  throw std::runtime_error("unreachable");
}

static void create_graph(SourceWriter &src, const SymbolicIR &symbolic_ir,
                         const ComputeGraph &compute_graph,
                         const ShaderRegistry &shader_registry,
                         const denox::dnx::Model *dnx,
                         std::vector<bool> &referenced_symbols,
                         std::string_view module_name) {

  const uint32_t n = compute_graph.nodes.size();
  std::vector<std::string> namespaces(n);
  for (uint32_t nid = 0; nid < n; ++nid) {
    const auto &node = compute_graph.nodes[nid];

    if (std::holds_alternative<ComputeDispatch>(node.op)) {
      const auto &compute_dispatch = std::get<ComputeDispatch>(node.op);
      std::string node_namespace = compute_dispatch.name;
      namespaces[nid] = node_namespace;
      // === Create push constant ====
      if (compute_dispatch.pc.size != 0) {
        const uint32_t pc_count = compute_dispatch.pc.fields.size();
        std::vector<PushConstantField> fields = compute_dispatch.pc.fields;
        std::ranges::sort(fields, [](const auto &lhs, const auto &rhs) {
          return lhs.offset < rhs.offset;
        });

        bool contigous_u32 = true;
        for (size_t i = 0; i < pc_count; ++i) {
          const auto &field = fields[i];
          if (field.type != PushConstantType::U32) {
            contigous_u32 = false;
            break;
          }
          if (field.offset != i * sizeof(uint32_t)) {
            contigous_u32 = false;
            break;
          }
        }
        if (contigous_u32) {
          std::string pcdef =
              fmt::format("const uint32_t {}_pc[{}] = {{", node_namespace,
                          compute_dispatch.pc.size / sizeof(uint32_t));
          bool first = true;
          for (const auto &field : fields) {
            if (!first) {
              pcdef.append(", ");
            }
            first = false;
            if (field.value.type == denox::dnx::ScalarSource_literal) {
              pcdef.append(
                  fmt::format("{}", access_symbol(symbolic_ir, field.value,
                                                  referenced_symbols)));
            } else {
              pcdef.append(fmt::format(
                  "(uint32_t)({})",
                  access_symbol(symbolic_ir, field.value, referenced_symbols)));
            }
          }
          pcdef.append("};");

          src.append(pcdef);

        } else {
          src.append(fmt::format("const uint8_t {}_pc[{}];", node_namespace,
                                 compute_dispatch.pc.size));
          src.append("{");
          src.push_indentation();
          for (uint32_t p = 0; p < pc_count; ++p) {
            const auto &pc = fields[p];
            if (pc.type != PushConstantType::I64) {
              src.append(fmt::format(
                  "const {} pc{} = ({}){};",
                  push_constant_type_to_string(pc.type), p,
                  push_constant_type_to_string(pc.type),
                  access_symbol(symbolic_ir, pc.value, referenced_symbols)));
            } else {
              src.append(fmt::format(
                  "const {} pc{} = {};", push_constant_type_to_string(pc.type),
                  p, access_symbol(symbolic_ir, pc.value, referenced_symbols)));
            }

            src.append(fmt::format("memcpy({}_pc + {}, &pc{}, sizeof({}));",
                                   node_namespace, pc.offset, p,
                                   push_constant_type_to_string(pc.type)));
          }
          src.pop_indentation();
          src.append("}");
        }
      }

      const auto &binary = shader_registry.binaries[compute_dispatch.binary_id];

      // === Create node ===
      src.append(fmt::format(
          "const int {}_id = dt_node_add(graph, module, \"{}\", \"{}\",",
          node_namespace, module_name, binary.name));
      src.push_indentation(2);
      src.append(fmt::format(
          "{} * DT_LOCAL_SIZE_X, {} * DT_LOCAL_SIZE_Y, {},",
          access_symbol(symbolic_ir, compute_dispatch.workgroup_count_x,
                        referenced_symbols),
          access_symbol(symbolic_ir, compute_dispatch.workgroup_count_y,
                        referenced_symbols),
          access_symbol(symbolic_ir, compute_dispatch.workgroup_count_z,
                        referenced_symbols)));

      if (compute_dispatch.pc.size != 0) {
        src.append(fmt::format("{}, (const int*){}_pc, {},",
                               compute_dispatch.pc.size, node_namespace,
                               node.sinksources.size()));
      } else {
        src.append(fmt::format("0, NULL, {},", node.sinksources.size()));
      }

      assert(!node.sinksources.empty());
      for (uint32_t i = 0; i < node.sinksources.size(); ++i) {
        const auto &sinksource = node.sinksources[i];
        std::string sinksource_desc = fmt::format(
            "\"{}\", \"{}\", \"{}\", \"{}\", &roi{}", sinksource.name,
            sinksource_type_to_string(sinksource.type),
            sinksource_chan_to_string(sinksource.chan),
            sinksource_format_to_string(sinksource.format),
            sinksource.buffer_roi_id);

        if (i == node.sinksources.size() - 1) {
          sinksource_desc += ");";
        } else {
          sinksource_desc.push_back(',');
        }
        src.append(sinksource_desc);
      }

      src.pop_indentation(2);
    } else if (std::holds_alternative<Upload>(node.op)) {
      const auto &upload = std::get<Upload>(node.op);
      namespaces[nid] = upload.name;

      src.append(
          fmt::format("int {}_id = dt_node_add(graph, module, \"{}\", \"{}\",",
                      upload.name, module_name, upload.name));
      src.push_indentation(2);
      src.append("1, 1, 1, 0, NULL, 1, ");
      for (uint32_t i = 0; i < node.sinksources.size(); ++i) {
        const auto &sinksource = node.sinksources[i];
        std::string sinksource_desc = fmt::format(
            "\"{}\", \"{}\", \"{}\", \"{}\", &roi{}", sinksource.name,
            sinksource_type_to_string(sinksource.type),
            sinksource_chan_to_string(sinksource.chan),
            sinksource_format_to_string(sinksource.format),
            sinksource.buffer_roi_id);

        if (i == node.sinksources.size() - 1) {
          sinksource_desc += ");";
        } else {
          sinksource_desc.push_back(',');
        }
        src.append(sinksource_desc);
      }
      src.pop_indentation(2);
    }
  }
  // Create connectors
  const uint32_t m = compute_graph.connectors.size();
  for (uint32_t i = 0; i < m; ++i) {
    const auto &connector = compute_graph.connectors[i];
    if (connector.src_node == external_sential) {
      assert(connector.dst_node != external_sential);
      const uint32_t input_index = connector.src_node_sinksource;
      fmt::println("input-index = {}", input_index);
      const auto &info = compute_graph.input_descriptors[input_index];
      src.append(fmt::format("if ({}_connector == NULL) {{", info.name));
      src.push_indentation();
      src.append(fmt::format(
          "dt_connector_copy(graph, module, {}_id, {}_id, {});", info.name,
          namespaces[connector.dst_node], connector.dst_node_sinksource));
      src.pop_indentation();
      src.append("} else {");
      src.push_indentation();
      src.append(fmt::format(
          "dt_node_connect_named(graph, {}_id, {}_connector, {}_id, \"{}\");",
          info.name, info.name, namespaces[connector.dst_node],
          compute_graph.nodes[connector.dst_node]
              .sinksources[connector.dst_node_sinksource]
              .name));
      src.pop_indentation();
      src.append("}");
    } else if (connector.dst_node == external_sential) {
      assert(connector.src_node != external_sential);
      const uint32_t output_index = connector.dst_node_sinksource;
      const auto &info = compute_graph.output_descriptors[output_index];
      src.append(fmt::format("if ({}_connector == NULL) {{", info.name));
      src.push_indentation();
      src.append(fmt::format(
          "dt_connector_copy(graph, module, {}_id, {}_id, {});", info.name,
          namespaces[connector.src_node], connector.src_node_sinksource));
      src.pop_indentation();
      src.append("} else {");
      src.push_indentation();
      src.append(fmt::format(
          "dt_node_connect_named(graph, {}_id, \"{}\", {}_id, {}_connector);",
          namespaces[connector.src_node],
          compute_graph.nodes[connector.src_node]
              .sinksources[connector.src_node_sinksource]
              .name,
          info.name, info.name));
      src.pop_indentation();
      src.append("}");
    } else {

      assert(connector.src_node != external_sential);
      assert(connector.src_node != external_sential);
      assert(connector.dst_node != none_sentinal);
      assert(connector.src_node != none_sentinal);
      assert(connector.dst_node != none_sentinal);
      assert(connector.src_node_sinksource != none_sentinal);
      assert(connector.dst_node_sinksource != none_sentinal);

      src.append(fmt::format(
          "dt_node_connect_named(graph, {}_id, \"{}\", {}_id, \"{}\");",
          namespaces[connector.src_node],
          compute_graph.nodes[connector.src_node]
              .sinksources[connector.src_node_sinksource]
              .name,
          namespaces[connector.dst_node],
          compute_graph.nodes[connector.dst_node]
              .sinksources[connector.dst_node_sinksource]
              .name));
    }
  }
}

} // namespace vkdt_denox

void vkdt_denox::def_func_denox_create_nodes(
    SourceWriter &src, const denox::dnx::Model *dnx,
    const SymbolicIR &symbolic_ir, const ShaderRegistry &shader_registery,
    const CompressedWeights &compresed_weights,
    const ComputeGraph &compute_graph, const std::string_view module_name) {
  src.add_include("stdint.h", IncludeType::System);
  src.add_include("string.h", IncludeType::System);
  src.add_include("stddef.h", IncludeType::System);
  src.add_include("modules/api.h", IncludeType::Local);

  std::string def =
      "static void denox_create_nodes(dt_graph_t* graph, dt_module_t* module";
  if (symbolic_ir.vars.empty()) {
    def.append(") {");
    src.append(def);
  } else {
    def.append(",");
    src.append(def);
    src.push_indentation(3);
    std::string valueParams = "";
    bool first = true;
    for (const auto &var : symbolic_ir.vars) {
      if (!first) {
        valueParams.append(", ");
      }
      first = false;
      valueParams.append(fmt::format("uint64_t {}", var));
    }
    valueParams.append(",");
    src.append(valueParams);

    assert(!compute_graph.input_descriptors.empty());
    for (const auto &input : compute_graph.input_descriptors) {
      src.append(fmt::format("int {}_id, const char* {}_connector,", input.name,
                             input.name));
    }
    assert(!compute_graph.output_descriptors.empty());
    first = true;
    for (size_t i = 0; i < compute_graph.output_descriptors.size(); ++i) {
      const auto &output = compute_graph.output_descriptors[i];

      if (i == compute_graph.output_descriptors.size() - 1) {
        src.append(fmt::format("int {}_id, const char* {}_connector) {{",
                               output.name, output.name));
      } else {
        src.append(fmt::format("int {}_id, const char* {}_connector,",
                               output.name, output.name));
      }
    }

    src.pop_indentation(3);
  }

  src.push_indentation();

  std::vector<bool> referenced_symbols(
      symbolic_ir.symir->ops()->size() + symbolic_ir.vars.size(), false);
  SourceWriter comp_src;

  create_buffer_rois(comp_src, symbolic_ir, compute_graph, referenced_symbols);
  create_graph(comp_src, symbolic_ir, compute_graph, shader_registery, dnx,
               referenced_symbols, module_name);

  SourceWriter sym_src;
  eval_symbolics(sym_src, symbolic_ir, referenced_symbols);

  src.append(sym_src.finish());
  src.append(comp_src.finish());

  src.pop_indentation();
  src.append("}");
}
