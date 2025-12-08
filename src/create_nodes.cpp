#include "create_nodes.hpp"
#include "compute_graph.hpp"
#include "shader_registry.hpp"
#include "source_writer.hpp"
#include "symbolics.hpp"
#include <atomic>
#include <dnx.h>
#include <fmt/base.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>

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
  }
  throw std::runtime_error("unreachable");
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

static size_t push_constant_type_bytesize(vkdt_denox::PushConstantType type) {
  switch (type) {
  case vkdt_denox::U32:
  case vkdt_denox::I32:
    return 4;
  case vkdt_denox::U16:
  case vkdt_denox::I16:
    return 2;
  case vkdt_denox::U64:
  case vkdt_denox::I64:
    return 8;
  }
  throw std::runtime_error("unreachable");
}

static std::string access_symbol(vkdt_denox::Symbol symbol,
                                 std::vector<uint32_t> &referenced_symbols) {
  if (symbol.type == denox::dnx::ScalarSource_symbolic) {
    const auto *sym_ref = static_cast<const denox::dnx::SymRef *>(symbol.ptr);
    ++referenced_symbols[sym_ref->sid()];
    return fmt::format("s{}", sym_ref->sid());
  } else if (symbol.type == denox::dnx::ScalarSource_literal) {
    return fmt::format(
        "{}", vkdt_denox::read_unsigned_scalar_literal(
                  static_cast<const denox::dnx::ScalarLiteral *>(symbol.ptr)));
  } else {
    throw std::runtime_error("invalid scalar source type!");
  }
}

static void eval_symbolics(vkdt_denox::SourceWriter &src,
                           const vkdt_denox::SymbolicIR &ir) {
  std::vector<bool> symbol_referenced;
}

void vkdt_denox::def_func_create_nodes(
    SourceWriter &src_file, const denox::dnx::Model *dnx,
    const SymbolicIR &symbolic_ir, const ShaderRegistry &shader_registry,
    const CompressedWeights &compressed_weights,
    const ComputeGraph &compute_graph, const std::string_view module_name) {

  src_file.add_include("stdint.h", IncludeType::System);
  src_file.add_include("string.h", IncludeType::System);
  src_file.add_include("stddef.h", IncludeType::System);
  src_file.add_include("modules/api.h", IncludeType::Local);

  src_file.append(
      "void create_nodes(dt_graph_t* graph, dt_module_t* module) {");
  src_file.push_indentation();

  std::vector<uint32_t> referenced_symbols(
      symbolic_ir.symir->var_count() + symbolic_ir.symir->ops()->size(), 0);
  // eval_symbolics(src, symbolic_ir);
  SourceWriter src_compute_graph;

  // ====== Create buffer-rois =======

  for (uint32_t i = 0; i < compute_graph.buffer_rois.size(); ++i) {
    const auto &buffer_roi = compute_graph.buffer_rois[i];
    if (buffer_roi.extent.has_value()) {
      throw std::runtime_error(
          "buffer-rois with extent are not implemented in the codegen.");
      // write extent semantics.
    } else if (std::holds_alternative<size_t>(buffer_roi.byte_size)) {
      size_t byte_size = std::get<size_t>(buffer_roi.byte_size);
      src_compute_graph.append(
          fmt::format("dt_roi_t roi{} = {{.wd = {}, .ht = 1}};", i, byte_size));
    } else if (std::holds_alternative<Symbol>(buffer_roi.byte_size)) {
      auto symbol = std::get<Symbol>(buffer_roi.byte_size);
      src_compute_graph.append(
          fmt::format("dt_roi_t roi{} = {{.wd = {}, .ht = 1}};", i,
                      access_symbol(symbol, referenced_symbols)));
    }
  }

  // ====== Create nodes ======

  const uint32_t node_count = compute_graph.nodes.size();
  for (uint32_t nid = 0; nid < node_count; ++nid) {
    const auto &node = compute_graph.nodes[nid];
    std::string node_namespace = fmt::format("n{}", nid);

    if (std::holds_alternative<ComputeDispatch>(node.op)) {
      const auto &compute_dispatch = std::get<ComputeDispatch>(node.op);

      if (compute_dispatch.pc.size != 0) {
        src_compute_graph.append(fmt::format(
            "uint8_t {}_pc[{}];", node_namespace, compute_dispatch.pc.size));
        src_compute_graph.append("{");
        src_compute_graph.push_indentation();
        const uint32_t pc_count = compute_dispatch.pc.fields.size();
        for (uint32_t p = 0; p < pc_count; ++p) {
          const auto &pc = compute_dispatch.pc.fields[p];
          if (pc.type != PushConstantType::I64) {
            src_compute_graph.append(
                fmt::format("const {} pc{} = ({}){};",
                            push_constant_type_to_string(pc.type), p,
                            push_constant_type_to_string(pc.type),
                            access_symbol(pc.value, referenced_symbols)));
          } else {
            src_compute_graph.append(fmt::format(
                "const {} pc{} = {};", push_constant_type_to_string(pc.type), p,
                access_symbol(pc.value, referenced_symbols)));
          }

          src_compute_graph.append(fmt::format(
              "memcpy({}_pc + {}, &pc{}, {});", node_namespace, pc.offset, p,

              push_constant_type_bytesize(pc.type)));
        }
        src_compute_graph.pop_indentation();
        src_compute_graph.append("}");
      }

      const auto &binary = shader_registry.binaries[compute_dispatch.binary_id];

      src_compute_graph.append(
          fmt::format("int {}_id = dt_node_add(", node_namespace));
      src_compute_graph.push_indentation(2);
      src_compute_graph.append(fmt::format("graph, module, \"{}\", \"{}\", ",
                                           module_name, binary.name));
      src_compute_graph.append(fmt::format(
          "{}, {}, {},",
          access_symbol(compute_dispatch.workgroup_count_x, referenced_symbols),
          access_symbol(compute_dispatch.workgroup_count_y, referenced_symbols),
          access_symbol(compute_dispatch.workgroup_count_z,
                        referenced_symbols)));
      if (compute_dispatch.pc.size != 0) {
        src_compute_graph.append(
            fmt::format("{}, (const int*){}_pc, {},", compute_dispatch.pc.size,
                        node_namespace, node.sinksources.size()));
      } else {
        src_compute_graph.append(
            fmt::format("0, NULL, {},", node.sinksources.size()));
      }

      for (uint32_t i = 0; i < node.sinksources.size(); ++i) {
        const auto &sinksource = node.sinksources[i];
        std::string sinksource_desc = fmt::format(
            "\"{}\", \"{}\", \"{}\", \"{}\", roi{}", sinksource.name,
            sinksource_type_to_string(sinksource.type),
            sinksource_chan_to_string(sinksource.chan),
            sinksource_format_to_string(sinksource.format),
            sinksource.buffer_roi_id);

        if (i == node.sinksources.size() - 1) {
          sinksource_desc += ");";
        } else {
          sinksource_desc.push_back(',');
        }
        src_compute_graph.append(sinksource_desc);
      }

      src_compute_graph.pop_indentation(2);

    } else if (std::holds_alternative<Upload>(node.op)) {
      const auto &upload = std::get<Upload>(node.op);

      src_compute_graph.append(
          fmt::format("int {}_id = dt_node_add(", node_namespace));
      src_compute_graph.push_indentation(2);
      src_compute_graph.append(fmt::format("graph, module, \"{}\", \"{}\", ",
                                           module_name, upload.name));
      src_compute_graph.append("1, 1, 1, 0, NULL, 1, ");
      for (uint32_t i = 0; i < node.sinksources.size(); ++i) {
        const auto &sinksource = node.sinksources[i];
        std::string sinksource_desc = fmt::format(
            "\"{}\", \"{}\", \"{}\", \"{}\", roi{}", sinksource.name,
            sinksource_type_to_string(sinksource.type),
            sinksource_chan_to_string(sinksource.chan),
            sinksource_format_to_string(sinksource.format),
            sinksource.buffer_roi_id);

        if (i == node.sinksources.size() - 1) {
          sinksource_desc += ");";
        } else {
          sinksource_desc.push_back(',');
        }
        src_compute_graph.append(sinksource_desc);
      }
      src_compute_graph.pop_indentation(2);
    }
  }

  for (uint32_t i = 0; i < compute_graph.connectors.size(); ++i) {
    const auto &connector = compute_graph.connectors[i];

    if (connector.src_node == input_sential) {
      src_compute_graph.append("//TODO: input connector");
    } else {
      assert(connector.src_node != none_sentinal);
      assert(connector.dst_node != none_sentinal);
      assert(connector.src_node_sinksource != none_sentinal);
      assert(connector.dst_node_sinksource != none_sentinal);

      src_compute_graph.append(fmt::format(
          "dt_node_connect_named(graph, n{}_id, \"{}\", n{}_id, \"{}\");",
          connector.src_node,
          compute_graph.nodes[connector.src_node]
              .sinksources[connector.src_node_sinksource]
              .name,
          connector.dst_node,
          compute_graph.nodes[connector.dst_node]
              .sinksources[connector.dst_node_sinksource]
              .name));
    }
  }

  src_compute_graph.append("// TODO: output connectors");

  // Finally generate symbolic eval code.

  SourceWriter src_sym;
  const uint32_t symvar_count = symbolic_ir.symir->var_count();

  std::vector<std::string> symbolic_expressions;

  for (uint32_t i = 0; i < symbolic_ir.sym_sources.size(); ++i) {
    const uint32_t sid = i;
    const auto &symsrc = symbolic_ir.sym_sources[i];
    const auto *tensor_info = dnx->inputs()->Get(symsrc.input_index);
    switch (symsrc.dim) {
    case SymbolicVarSourceDim::Height: {
      symbolic_expressions.push_back(
          fmt::format("int64_t s{} = module->connector[{}].roi.wd;", sid, symsrc.input_index));
      break;
    }
    case SymbolicVarSourceDim::Width: {
      symbolic_expressions.push_back(
          fmt::format("int64_t s{} = module->connector[{}].roi.ht;", sid, symsrc.input_index));
      break;
    }
    }
  }

  for (uint32_t i = 0; i < symbolic_ir.symir->ops()->size(); ++i) {
    uint32_t sid = i + symbolic_ir.symir->var_count();
    const auto *op = symbolic_ir.symir->ops()->Get(i);
    const auto opcode = op->opcode();

    std::string lhs;
    if (opcode & denox::dnx::SymIROpCode_LHSC) {
      lhs = fmt::format("{}", op->lhs());
    } else {
      lhs = fmt::format("s{}", op->lhs());
      ++referenced_symbols[op->lhs()];
    }

    std::string rhs;
    if (opcode & denox::dnx::SymIROpCode_RHSC) {
      rhs = fmt::format("{}", op->rhs());
    } else {
      rhs = fmt::format("s{}", op->rhs());
      ++referenced_symbols[op->rhs()];
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

    symbolic_expressions.push_back(fmt::format("int64_t s{} = {};", sid, expr));
  }

  std::vector<bool> pruned_expressions(symbolic_expressions.size(), false);

  // pruning step. (micro optimization, to remove unused branches)
  while (true) {
    bool pruned_once = false;

    for (uint32_t i = symvar_count; i < symbolic_expressions.size(); ++i) {
      if (pruned_expressions[i]) {
        continue;
      }
      if (referenced_symbols[i] > 0) {
        continue;
      }
      pruned_expressions[i] = true;
      pruned_once = true;
      const auto* op = symbolic_ir.symir->ops()->Get(i - symvar_count);
      if (!(op->opcode() & denox::dnx::SymIROpCode_LHSC)) {
        uint32_t sid = op->lhs();
        assert(referenced_symbols[sid] > 0);
        --referenced_symbols[sid];
      }

      if (!(op->opcode() & denox::dnx::SymIROpCode_RHSC)) {
        uint32_t sid = op->rhs();
        assert(referenced_symbols[sid] > 0);
        --referenced_symbols[sid];
      }
    }

    if (!pruned_once) {
      break;
    }
  }

  for (uint32_t i = 0; i < symbolic_expressions.size(); ++i) {
    if (referenced_symbols[i] > 0) {
      src_sym.append(symbolic_expressions[i]);
    }
  }

  src_file.append(src_sym.finish());
  src_file.append(src_compute_graph.finish());
  src_file.pop_indentation();
  src_file.append("}");
}
