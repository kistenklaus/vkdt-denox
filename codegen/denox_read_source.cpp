#include "denox_read_source.hpp"
#include "compress_weights.hpp"
#include "source_writer.hpp"
#include <variant>

void vkdt_denox::def_func_denox_read_source(
    SourceWriter &src, const ComputeGraph &compute_graph,
    const CompressedWeights &compressed_weights, std::string_view weights_path,
    std::string_view module_name) {
  src.add_include("stdint.h", IncludeType::System);
  src.add_include("stdio.h", IncludeType::System);
  src.add_include("modules/api.h", IncludeType::Local);

  src.append("static int denox_read_source(dt_module_t* mod, void* mapped, "
             "dt_read_source_params_t* p) {");
  src.push_indentation();
  bool first = true;
  for (const auto &node : compute_graph.nodes) {
    if (!std::holds_alternative<Upload>(node.op)) {
      continue;
    }
    const Upload &upload = std::get<Upload>(node.op);

    if (first) {
      src.append(fmt::format("if (p->node->kernel == dt_token(\"{}\")) {{",
                             upload.name));
    } else {
      src.append(fmt::format("}} else (p->node->kernel == dt_token(\"{}\")) {{",
                             upload.name));
    }

    src.push_indentation();
    src.append(fmt::format(
        "FILE* f = dt_graph_open_resource(mod->graph, 0, \"{}\", \"rb\");",
        weights_path));
    src.append("if (!f) {");
    src.push_indentation();
    src.append("snprintf(mod->graph->gui_msg_buf, "
               "sizeof(mod->graph->gui_msg_buf),");
    src.push_indentation(3);
    src.append(fmt::format("\"{}: could not find \\\"{}\\\"\");", module_name,
                           weights_path));
    src.pop_indentation(3);
    src.append("return 1;");
    src.pop_indentation();
    src.append("}");

    src.append("fseek(f, 0, SEEK_END);");
    src.append("const size_t size = ftell(f);");
    src.append(fmt::format("const size_t expected_size = {};",
                           compressed_weights.data.size()));

    src.append("if (size != expected_size) {");
    src.push_indentation();
    src.append(
        "snprintf(mod->graph->gui_msg_buf, sizeof(mod->graph->gui_msg_buf),");
    src.push_indentation(3);
    src.append(
        fmt::format("\"{}: weight file \\\"{}\\\" has unexpected size!\");",
                    module_name, weights_path));
    src.pop_indentation(3);
    src.append("fclose(f);");
    src.append("return 1;");

    src.pop_indentation();
    src.append("}");

    src.append("fseek(f, 0, SEEK_SET);");
    src.append("fread(mapped, size, 1, f);");
    src.append("fclose(f);");

    src.pop_indentation();
    src.append("}");
  }
  src.append("return 0;");

  src.pop_indentation();
  src.append("}");
}
