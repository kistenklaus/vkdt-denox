#include "create_nodes.hpp"
#include "source_writer.hpp"

void vkdt_denox::def_func_create_nodes(SourceWriter &src,
                                       const denox::dnx::Model *dnx) {

  src.add_include("modules/api", IncludeType::Local);

  src.append("void create_nodes(dt_graph_t* graph, dt_module_t* module) {");
  src.push_indentation();

  uint32_t input_count = 0;
  src.append(fmt::format("TensorExtent input_extents[{}];", input_count));

  // TODO: Parse extents of input connectors.
  src.append("{");
  src.push_indentation();
  src.append("//TODO: Parse tensor extents from module or graph.");
  src.append("input_extents[0].width = 1920; // <- hardcoded");
  src.append("input_extents[0].height = 1080; // <- hardcoded");
  src.pop_indentation();
  src.append("}");

  src.append("SymbolTable symbol_table;");
  src.append("eval_symbolic_expressions(&symbol_table, input_extents);");

  src.append("//TODO: create nodes (each node is a single dispatch");

  src.pop_indentation();
  src.append("}");
}
