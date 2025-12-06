#include "tensor_extent.hpp"
#include "source_writer.hpp"

void vkdt_denox::def_struct_tensor_extent(SourceWriter &src) {
  src.add_include("stdint.h", IncludeType::System);

  src.append("struct TensorExtent {");
  src.push_indentation();
  src.append("uint32_t width;");
  src.append("uint32_t height;");
  src.pop_indentation();
  src.append("};");
}

