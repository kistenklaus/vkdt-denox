#include "symbolics.hpp"
#include "source_writer.hpp"
#include <dnx.h>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
#include <stdexcept>

static void register_symbol_as_referenced(vkdt_denox::SymbolicIR &ir,
                                          denox::dnx::ScalarSource type,
                                          const void *data) {
  if (type == denox::dnx::ScalarSource_symbolic) {
    auto *symRef = static_cast<const denox::dnx::SymRef *>(data);
    uint32_t sid = symRef->sid();
    if (sid >= ir.symbol_is_referrenced.size()) {
      throw std::runtime_error("invalid symbol ref");
    }
    ir.symbol_is_referrenced[sid] = true;
  }
}

static void register_all_referenced_symbols(vkdt_denox::SymbolicIR &ir,
                                            const denox::dnx::Model *dnx) {
  { // 1. Tensors
    const uint32_t tensor_count = dnx->tensors()->size();
    for (uint32_t i = 0; i < tensor_count; ++i) {
      const auto *tensor = dnx->tensors()->Get(i);
      register_symbol_as_referenced(ir, tensor->size_type(), tensor->size());
      register_symbol_as_referenced(ir, tensor->offset_type(),
                                    tensor->offset());
    }
  }

  { // 2. Buffers
    const uint32_t buffer_count = dnx->buffers()->size();
    for (uint32_t i = 0; i < buffer_count; ++i) {
      const auto *buffer = dnx->buffers()->Get(i);
      register_symbol_as_referenced(ir, buffer->size_type(), buffer->size());
    }
  }

  { // 3. Dispatches
    const uint32_t dispatch_count = dnx->dispatches()->size();
    for (uint32_t i = 0; i < dispatch_count; ++i) {
      const auto *dispatch = dnx->dispatches()->Get(i);
      const auto dispatch_type = dnx->dispatches_type()->Get(i);
      if (dispatch_type == denox::dnx::Dispatch_ComputeDispatch) {
        const auto *compute_dispatch =
            static_cast<const denox::dnx::ComputeDispatch *>(dispatch);
        register_symbol_as_referenced(
            ir, compute_dispatch->workgroup_count_x_type(),
            compute_dispatch->workgroup_count_x());
        register_symbol_as_referenced(
            ir, compute_dispatch->workgroup_count_y_type(),
            compute_dispatch->workgroup_count_y());
        register_symbol_as_referenced(
            ir, compute_dispatch->workgroup_count_z_type(),
            compute_dispatch->workgroup_count_z());
        const auto *push_constant_fields =
            compute_dispatch->push_constant()->fields();
        const uint32_t push_constant_count = push_constant_fields->size();
        for (uint32_t i = 0; i < push_constant_count; ++i) {
          const auto *push_constant_field = push_constant_fields->Get(i);
          register_symbol_as_referenced(ir, push_constant_field->source_type(),
                                        push_constant_field->source());
        }
      } else {
        throw std::runtime_error("not implemented");
      }
    }
  }

  { // 4. Inputs
    const uint32_t input_count = dnx->inputs()->size();
    for (uint32_t i = 0; i < input_count; ++i) {
      const auto *tensor_info = dnx->inputs()->Get(i);
      register_symbol_as_referenced(ir, tensor_info->width_type(),
                                    tensor_info->width());
      register_symbol_as_referenced(ir, tensor_info->height_type(),
                                    tensor_info->height());
      register_symbol_as_referenced(ir, tensor_info->channels_type(),
                                    tensor_info->channels());
    }
  }
  { // 5. Outputs
    const uint32_t output_count = dnx->outputs()->size();
    for (uint32_t i = 0; i < output_count; ++i) {
      const auto *tensor_info = dnx->outputs()->Get(i);
      register_symbol_as_referenced(ir, tensor_info->width_type(),
                                    tensor_info->width());
      register_symbol_as_referenced(ir, tensor_info->height_type(),
                                    tensor_info->height());
      register_symbol_as_referenced(ir, tensor_info->channels_type(),
                                    tensor_info->channels());
    }
  }

  // 6. ValueNames
  {
    const uint32_t value_name_count = dnx->value_names()->size();
    for (uint32_t i = 0; i < value_name_count; ++i) {
      const auto *value_name = dnx->value_names()->Get(i);
      register_symbol_as_referenced(ir, value_name->value_type(),
                                    value_name->value());
    }
  }
}

vkdt_denox::SymbolicIR
vkdt_denox::read_symbolic_ir(const denox::dnx::Model *dnx) {
  SymbolicIR ir;
  const uint32_t var_count = dnx->sym_ir()->var_count();
  const uint32_t symbol_count = dnx->sym_ir()->ops()->size() + var_count;

  // 1. Search for all symbols that are even used anywhere.
  ir.symbol_is_referrenced.resize(symbol_count);
  register_all_referenced_symbols(ir, dnx);

  ir.symir = dnx->sym_ir();

  const uint32_t input_count = dnx->inputs()->size();
  ir.sym_sources.resize(var_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    const auto *tensor_info = dnx->inputs()->Get(i);
    if (tensor_info->width_type() == denox::dnx::ScalarSource_symbolic) {
      uint32_t sid = tensor_info->width_as_symbolic()->sid();
      if (sid < var_count) {
        ir.sym_sources[sid].dim = SymbolicVarSourceDim::Width;
        ir.sym_sources[sid].input_index = i;
      }
    }
    if (tensor_info->height_type() == denox::dnx::ScalarSource_symbolic) {
      uint32_t sid = tensor_info->height_as_symbolic()->sid();
      if (sid < var_count) {
        ir.sym_sources[sid].dim = SymbolicVarSourceDim::Height;
        ir.sym_sources[sid].input_index = i;
      }
    }
  }

  return ir;
}

// void vkdt_denox::def_struct_symbolic_table(vkdt_denox::SourceWriter &src,
//                                            SymbolicIR &ir) {
//   src.add_include("stdint.h", IncludeType::System);
//
//   src.append("struct SymbolTable {");
//   src.push_indentation();
//
//   for (uint32_t i = 0; i < ir.symbol_is_referrenced.size(); ++i) {
//     if (ir.symbol_is_referrenced[i]) {
//       src.append(fmt::format("int64_t r{};", i));
//     }
//   }
//   src.pop_indentation();
//   src.append("};");
// }
//
// std::string vkdt_denox::access_symbol(const SymbolicIR &ir,
//                                       std::string_view table_ptr_var,
//                                       Symbol symbol, bool deref) {
//   if (symbol.type == denox::dnx::ScalarSource_symbolic) {
//     uint32_t sid = static_cast<const denox::dnx::SymRef *>(symbol.ptr)->sid();
//     if (!ir.symbol_is_referrenced[sid]) {
//       throw std::runtime_error(fmt::format("Trying to access unreferred symbol [sid={}]", sid));
//     }
//     if (deref) {
//       return fmt::format("{}->r{}", table_ptr_var, sid);
//     } else {
//       return fmt::format("{}.r{}", table_ptr_var, sid);
//     }
//   } else {
//     const auto *literal =
//         static_cast<const denox::dnx::ScalarLiteral *>(symbol.ptr);
//     int64_t v;
//     switch (literal->dtype()) {
//     case denox::dnx::ScalarType_I16: {
//       int16_t tmp;
//       std::memcpy(&tmp, literal->bytes()->data(), sizeof(int16_t));
//       v = static_cast<int64_t>(tmp);
//       break;
//     }
//     case denox::dnx::ScalarType_U16: {
//       uint16_t tmp;
//       std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint16_t));
//       v = static_cast<int64_t>(tmp);
//       break;
//     }
//     case denox::dnx::ScalarType_I32: {
//       int32_t tmp;
//       std::memcpy(&tmp, literal->bytes()->data(), sizeof(int32_t));
//       v = static_cast<int64_t>(tmp);
//       break;
//     }
//     case denox::dnx::ScalarType_U32: {
//       uint32_t tmp;
//       std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint32_t));
//       v = static_cast<int64_t>(tmp);
//       break;
//     }
//     case denox::dnx::ScalarType_I64: {
//       std::memcpy(&v, literal->bytes()->data(), sizeof(int64_t));
//       break;
//     }
//     case denox::dnx::ScalarType_U64: {
//       uint64_t tmp;
//       std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint64_t));
//       v = static_cast<int64_t>(tmp);
//       break;
//     }
//     case denox::dnx::ScalarType_F16:
//     case denox::dnx::ScalarType_F32:
//     case denox::dnx::ScalarType_F64:
//       throw std::runtime_error("Floating point types, are not allowed in "
//                                "scalar sources. Invalid dnx file format!");
//       break;
//     }
//     return fmt::format("{}", v);
//   }
// }
//
// void vkdt_denox::def_func_eval_symbolic_expressions(
//     vkdt_denox::SourceWriter &src, SymbolicIR &ir) {
//
//   src.add_include("stdint.h", IncludeType::System);
//   src.append(
//       "static void eval_symbolic_expressions(struct SymbolTable* table, struct TensorExtent* "
//       "inputExtents) {");
//   src.push_indentation();
//   for (uint32_t sid = 0; sid < ir.sym_sources.size(); ++sid) {
//     const auto &source = ir.sym_sources[sid];
//     std::string varsource;
//     if (source.dim == SymbolicVarSourceDim::Width) {
//       varsource = fmt::format("inputExtents[{}].width", source.input_index);
//     } else if (source.dim == SymbolicVarSourceDim::Height) {
//       varsource = fmt::format("inputExtents[{}].height", source.input_index);
//     } else {
//       throw std::runtime_error("unreachable");
//     }
//     if (ir.symbol_is_referrenced[sid]) {
//       src.append(fmt::format("table->r{} = {};", sid, varsource));
//     } else {
//       src.append(fmt::format("uint64_t r{} = {}", sid, varsource));
//     }
//   }
//
//   const uint32_t opcount = ir.symir->ops()->size();
//   const uint32_t varcount = ir.symir->var_count();
//   for (uint32_t i = 0; i < opcount; ++i) {
//     uint32_t sid = i + varcount;
//     std::string expr;
//     const auto *op = ir.symir->ops()->Get(i);
//     denox::dnx::SymIROpCode opcode = op->opcode();
//
//     const bool lhsc = opcode & denox::dnx::SymIROpCode_LHSC;
//     std::string lhs;
//     if (lhsc) {
//       lhs = fmt::format("{}", op->lhs());
//     } else {
//       uint32_t lhs_sid = static_cast<uint32_t>(op->lhs());
//       if (ir.symbol_is_referrenced[lhs_sid]) {
//         lhs = fmt::format("table->r{}", lhs_sid);
//       } else {
//         lhs = fmt::format("r{}", lhs_sid);
//       }
//     }
//
//     const bool rhsc = opcode & denox::dnx::SymIROpCode_RHSC;
//     std::string rhs;
//     if (rhsc) {
//       rhs = fmt::format("{}", op->rhs());
//     } else {
//       uint32_t rhs_sid = static_cast<uint32_t>(op->rhs());
//       if (ir.symbol_is_referrenced[rhs_sid]) {
//         rhs = fmt::format("table->r{}", rhs_sid);
//       } else {
//         rhs = fmt::format("r{}", rhs_sid);
//       }
//     }
//
//     std::string res;
//     if (ir.symbol_is_referrenced[sid]) {
//       res = fmt::format("table->r{}", sid);
//     } else {
//       res = fmt::format("int64_t r{}", sid);
//     }
//
//     // SymIROpCode_ADD = 1,
//     // SymIROpCode_SUB = 2,
//     // SymIROpCode_MUL = 3,
//     // SymIROpCode_DIV = 4,
//     // SymIROpCode_MOD = 5,
//     // SymIROpCode_MIN = 6,
//     // SymIROpCode_MAX = 7,
//     if (opcode & denox::dnx::SymIROpCode_ADD) {
//       src.append(fmt::format("{} = {} + {};", res, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_SUB) {
//       src.append(fmt::format("{} = {} - {};", res, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_MUL) {
//       src.append(fmt::format("{} = {} * {};", res, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_DIV) {
//       src.append(fmt::format("{} = {} / {};", res, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_MOD) {
//       src.append(fmt::format("{} = {} % {};", res, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_MIN) {
//       src.append(
//           fmt::format("{} = {} < {} ? {} : {};", res, lhs, rhs, lhs, rhs));
//     } else if (opcode & denox::dnx::SymIROpCode_MAX) {
//       src.append(
//           fmt::format("{} = {} < {} ? {} : {};", res, lhs, rhs, rhs, lhs));
//     }
//   }
//
//   src.pop_indentation();
//
//   src.append("}");
// }

uint64_t vkdt_denox::read_unsigned_scalar_literal(
    const denox::dnx::ScalarLiteral *literal) {
  uint64_t v;
  switch (literal->dtype()) {
  case denox::dnx::ScalarType_I16: {
    int16_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int16_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U16: {
    uint16_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint16_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_I32: {
    int32_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int32_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U32: {
    uint32_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint32_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_I64: {
    int64_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(int64_t));
    v = static_cast<uint64_t>(tmp);
    break;
  }
  case denox::dnx::ScalarType_U64: {
    uint64_t tmp;
    std::memcpy(&tmp, literal->bytes()->data(), sizeof(uint64_t));
    v = tmp;
    break;
  }
  case denox::dnx::ScalarType_F16:
  case denox::dnx::ScalarType_F32:
  case denox::dnx::ScalarType_F64:
    throw std::runtime_error("Floating point types, are not allowed in "
                             "scalar sources. Invalid dnx file format!");
    break;
  }
  return v;
}
