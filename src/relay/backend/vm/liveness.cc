#include "liveness.h"

#include <tvm/runtime/vm.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

using namespace runtime::vm;

struct OpcodeHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<Index>(t);
  }
};

inline size_t GetAllocSize(const Instruction& instr) {
  CHECK(instr.op == Opcode::AllocTensor || instr.op == Opcode::AllocStorage);
  if (instr.op == Opcode::AllocStorage) return instr.alloc_storage.allocation_size;
  size_t size = 1;
  for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
    size *= static_cast<size_t>(instr.alloc_tensor.shape[i]);
  }
  size *= (instr.alloc_tensor.dtype.bits * instr.alloc_tensor.dtype.lanes + 7) / 8;
  return size;
}

// The size of each alloctensor
std::vector<std::pair<Instruction, size_t>> MemoryCadidates(
    const std::vector<Instruction>& instructions) {
  std::vector<std::pair<Instruction, size_t>> ret;
  for (const auto& it : instructions) {
    if (it.op == Opcode::AllocTensor) {
      ret.push_back(std::make_pair(it, GetAllocSize(it)));
    }
  }
  // auto cmp = [](const std::pair<Instruction, size_t>& a, const std::pair<Instruction, size_t>& b) {
  //   return a.second >= b.second;
  // };
  // std::sort(ret.begin(), ret.end(), cmp);
  return ret;
}

std::vector<runtime::vm::Instruction> ShuffleConstStorage(
    const std::vector<runtime::vm::Instruction>& instructions) {
  std::vector<runtime::vm::Instruction> ret;
  for (const auto& it : instructions) {
    if (it.op == runtime::vm::Opcode::LoadConst) {
      ret.push_back(it);
    }
  }

  for (const auto& it : instructions) {
    if (it.op == runtime::vm::Opcode::AllocStorage) {
      ret.push_back(it);
    }
  }

  for (const auto& it : instructions) {
    if (it.op != runtime::vm::Opcode::LoadConst && it.op != runtime::vm::Opcode::AllocStorage) {
      ret.push_back(it);
    }
  }
  return ret;
}

void Replace(Index reg, Index num_arg, std::unordered_map<Index, Index>& replaced, Index& idx) {
  if (replaced.count(reg) == 0 && reg >= num_arg) {
    replaced[reg] = idx++;
  }
}

std::vector<runtime::vm::Instruction> UpdateRegisterId(
    const std::vector<runtime::vm::Instruction>& instructions, size_t num_args) {
  Index idx = num_args;
  std::unordered_map<Index, Index> replaced;
  std::vector<Instruction> updated;
  RegName dst = -1;
  std::unordered_set<Opcode, OpcodeHash> skip = {Opcode::Fatal, Opcode::InvokePacked, Opcode::If,
                                                 Opcode::Ret};
  for (const auto& instr : instructions) {
    if (skip.count(instr.op) == 0) {
      Replace(instr.dst, num_args, replaced, idx);
      dst = replaced[instr.dst];
    }
    switch (instr.op) {
      case Opcode::Move: {
        Replace(instr.from, num_args, replaced, idx);
        updated.push_back(Instruction::Move(replaced[instr.from], dst));
        break;
      }
      case Opcode::Invoke: {
        std::vector<RegName> args;
        for (Index i = 0; i < instr.num_args; ++i) {
          Replace(instr.invoke_args_registers[i], num_args, replaced, idx);
          args.push_back(replaced[instr.invoke_args_registers[i]]);
        }
        updated.push_back(Instruction::Invoke(instr.func_index, args, dst));
        break;
      }
      case Opcode::InvokePacked: {
        std::vector<RegName> args;
        for (Index i = 0; i < instr.arity; ++i) {
          Replace(instr.packed_args[i], num_args, replaced, idx);
          args.push_back(replaced[instr.packed_args[i]]);
        }
        updated.push_back(
            Instruction::InvokePacked(instr.packed_index, instr.arity, instr.output_size, args));
        break;
      }
      case Opcode::InvokeClosure: {
        Replace(instr.closure, num_args, replaced, idx);

        std::vector<RegName> args;
        for (Index i = 0; i < instr.num_closure_args; ++i) {
          Replace(instr.closure_args[i], num_args, replaced, idx);
          args.push_back(replaced[instr.closure_args[i]]);
        }
        updated.push_back(Instruction::InvokeClosure(replaced[instr.closure], args, dst));
        break;
      }
      case Opcode::GetField: {
        Replace(instr.object, num_args, replaced, idx);
        updated.push_back(Instruction::GetField(replaced[instr.object], instr.field_index, dst));
        break;
      }
      case Opcode::GetTag: {
        Replace(instr.object, num_args, replaced, idx);
        updated.push_back(Instruction::GetTag(replaced[instr.object], dst));
        break;
      }
      case Opcode::If: {
        Replace(instr.if_op.test, num_args, replaced, idx);
        Replace(instr.if_op.target, num_args, replaced, idx);

        updated.push_back(Instruction::If(replaced[instr.if_op.test], replaced[instr.if_op.target],
                                          instr.if_op.true_offset, instr.if_op.false_offset));
        break;
      }
      case Opcode::AllocTensor: {
        Replace(instr.alloc_tensor.storage, num_args, replaced, idx);
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }
        updated.push_back(Instruction::AllocTensor(replaced[instr.alloc_tensor.storage], shape,
                                                   instr.alloc_tensor.dtype, dst));
        break;
      }
      case Opcode::AllocTensorReg: {
        Replace(instr.alloc_tensor_reg.storage, num_args, replaced, idx);
        Replace(instr.alloc_tensor_reg.shape_register, num_args, replaced, idx);

        updated.push_back(Instruction::AllocTensorReg(
            replaced[instr.alloc_tensor_reg.storage],
            replaced[instr.alloc_tensor_reg.shape_register], instr.alloc_tensor_reg.dtype, dst));
        break;
      }
      case Opcode::AllocADT: {
        std::vector<RegName> fields;
        for (Index i = 0; i < instr.num_fields; ++i) {
          Replace(instr.datatype_fields[i], num_args, replaced, idx);
          fields.push_back(replaced[instr.datatype_fields[i]]);
        }
        updated.push_back(
            Instruction::AllocADT(instr.constructor_tag, instr.num_fields, fields, dst));
        break;
      }
      case Opcode::AllocClosure: {
        std::vector<RegName> free_vars;
        for (Index i = 0; i < instr.num_freevar; ++i) {
          Replace(instr.free_vars[i], num_args, replaced, idx);
          free_vars.push_back(replaced[instr.free_vars[i]]);
        }
        updated.push_back(
            Instruction::AllocClosure(instr.func_index, instr.num_freevar, free_vars, dst));
        break;
      }
      case Opcode::AllocStorage: {
        Replace(instr.alloc_storage.allocation_size, num_args, replaced, idx);
        Replace(instr.alloc_storage.alignment, num_args, replaced, idx);
        updated.push_back(Instruction::AllocStorage(replaced[instr.alloc_storage.allocation_size],
                                                    replaced[instr.alloc_storage.alignment],
                                                    instr.alloc_storage.dtype_hint, dst));
        break;
      }
      case Opcode::Ret: {
        Replace(instr.result, num_args, replaced, idx);
        updated.push_back(Instruction::Ret(replaced[instr.result]));
        break;
      }
      case Opcode::LoadConst: {
        updated.push_back(Instruction::LoadConst(instr.const_index, dst));
        break;
      }
      default:
        updated.push_back(instr);
        LOG(WARNING) << "opcode: " << static_cast<Index>(instr.op) << " is not processed.";
        break;
    }
  }
  return updated;
}

inline void UpdateLiveRange(Index rid, Index pos, LiveRangeMap& intv_map) {
  if (intv_map.count(rid) == 0) {
    intv_map[rid] = Interval(pos, pos);
  } else{
    intv_map[rid].start = std::min(intv_map[rid].start, pos);
    intv_map[rid].end = std::max(intv_map[rid].end, pos);
  }
}

LiveRangeMap LiveRange(const std::vector<runtime::vm::Instruction>& instructions) {
  LiveRangeMap ret;
  for (size_t i = 0; i < instructions.size(); i++) {
    const auto& instr = instructions[i];
    switch (instr.op) {
      case Opcode::LoadConst: {
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::AllocTensor: {
        UpdateLiveRange(instr.alloc_tensor.storage, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::AllocStorage: {
        UpdateLiveRange(instr.alloc_storage.allocation_size, i, ret);
        UpdateLiveRange(instr.alloc_storage.alignment, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::InvokePacked: {
        for (Index arity = 0; arity < instr.arity; ++arity) {
          UpdateLiveRange(instr.packed_args[arity], i, ret);
        }
        break;
      }
      case Opcode::Invoke: {
        for (Index arg = 0; arg < instr.num_args; ++arg) {
          UpdateLiveRange(instr.invoke_args_registers[arg], i, ret);
        }
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::Ret: {
        UpdateLiveRange(instr.result, i, ret);
        break;
      }
      case Opcode::AllocTensorReg: {
        UpdateLiveRange(instr.alloc_tensor_reg.storage, i, ret);
        UpdateLiveRange(instr.alloc_tensor_reg.shape_register, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::AllocADT: {
        for (Index field = 0; field < instr.num_fields; ++field) {
          UpdateLiveRange(instr.datatype_fields[field], i, ret);
        }
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::AllocClosure: {
        for (Index var = 0; var < instr.num_freevar; ++var) {
          UpdateLiveRange(instr.free_vars[var], i, ret);
        }
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::InvokeClosure: {
        UpdateLiveRange(instr.closure, i, ret);
        for (Index arg = 0; arg < instr.num_closure_args; ++arg) {
          UpdateLiveRange(instr.closure_args[arg], i, ret);
        }
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::GetField: {
        UpdateLiveRange(instr.object, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::GetTag: {
        UpdateLiveRange(instr.object, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      case Opcode::If: {
        UpdateLiveRange(instr.if_op.test, i, ret);
        UpdateLiveRange(instr.if_op.target, i, ret);
        break;
      }
      case Opcode::Move: {
        UpdateLiveRange(instr.from, i, ret);
        UpdateLiveRange(instr.dst, i, ret);
        break;
      }
      default: {
        LOG(WARNING) << "opcode: " << static_cast<Index>(instr.op) << " is not processed.";
        break;
      }
    }
  }
  return ret;
}

Index FindFirstFitMemory(const LiveRangeMap& live_range,
                         const std::vector<std::pair<Instruction, size_t>>& memory_candidates,
                         const Instruction& cur_alloc, size_t pos) {
  CHECK(cur_alloc.op == Opcode::AllocTensor);
  size_t required = GetAllocSize(cur_alloc);
  for (size_t i = 0; i < memory_candidates.size(); i++) {
    const auto& instr = memory_candidates[i];
    CHECK_GE(live_range.count(instr.first.dst), 0U);
    Interval intv = live_range.at(instr.first.dst);
    CHECK(instr.first.op == Opcode::AllocTensor);
    if (instr.second >= required && static_cast<size_t>(intv.end) < pos) {
      return i;
    }
  }
  return -1;
}

Index UpdateRegisterUse(std::vector<runtime::vm::Instruction>& instructions, Index start,
                        Index target, Index replacement) {
  Index last_use = start;
  for (size_t idx = start; idx < instructions.size(); idx++) {
    Instruction& instr = instructions[idx];
    if (instr.dst == target) {
      instr.dst = replacement;
      last_use = idx;
    }

    switch (instr.op) {
      case Opcode::Invoke: {
        for (Index i = 0; i < instr.num_args; ++i) {
          if (instr.invoke_args_registers[i] == target) {
            instr.invoke_args_registers[i] = replacement;
            last_use = idx;
          }
        }
        break;
      }
      case Opcode::InvokePacked: {
        std::vector<RegName> args;
        for (Index i = 0; i < instr.arity; ++i) {
          if (instr.packed_args[i] == target) {
            instr.packed_args[i] = replacement;
            last_use = idx;
          }
        }
        break;
      }
      case Opcode::InvokeClosure: {
        if (instr.closure == target) {
          instr.closure = replacement;
          last_use = idx;
        }

        for (Index i = 0; i < instr.num_closure_args; ++i) {
          if (instr.closure_args[i] == target) {
            instr.closure_args[i] = replacement;
            last_use = idx;
          }
        }
        break;
      }
      case Opcode::GetField: {
        if (instr.object == target) {
          instr.object = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::GetTag: {
        if (instr.object == target) {
          instr.object = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::If: {
        if (instr.if_op.test == target) {
          instr.if_op.test = replacement;
          last_use = idx;
        }
        if (instr.if_op.target == target) {
          instr.if_op.target = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::AllocTensor: {
        if (instr.alloc_tensor.storage == target) {
          instr.alloc_tensor.storage = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::AllocTensorReg: {
        if (instr.alloc_tensor_reg.storage == target) {
          instr.alloc_tensor_reg.storage = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::AllocADT: {
        for (Index i = 0; i < instr.num_fields; ++i) {
          if (instr.datatype_fields[i] == target) {
            instr.datatype_fields[i] = replacement;
            last_use = idx;
          }
        }
        break;
      }
      case Opcode::AllocClosure: {
        for (Index i = 0; i < instr.num_freevar; ++i) {
          if (instr.free_vars[i] == target) {
            instr.free_vars[i] = replacement;
            last_use = idx;
          }
        }
        break;
      }
      case Opcode::AllocStorage: {
        if (instr.alloc_storage.allocation_size == target) {
          instr.alloc_storage.allocation_size = replacement;
          last_use = idx;
        }
        if (instr.alloc_storage.alignment == target) {
          instr.alloc_storage.alignment = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::Ret: {
        if (instr.result == target) {
          instr.result = replacement;
          last_use = idx;
        }
        break;
      }
      case Opcode::LoadConst:
        break;
      default:
        LOG(WARNING) << "opcode: " << static_cast<Index>(instr.op) << " is not processed.";
        break;
    }
  }
  return last_use;
}

// Cleanup redundant memory
std::vector<Instruction> Cleanup(const std::map<size_t, Index>& remove_map,
                                 std::vector<Instruction>& instructions,
                                 LiveRangeMap& live_range) {
  std::unordered_set<size_t> redundant_alloc_storage;
  for (auto it = remove_map.rbegin(); it != remove_map.rend(); ++it) {
    CHECK_GE(it->first, 0U);
    CHECK_LT(it->first, instructions.size());
    const auto& instr = instructions[it->first];
    CHECK(instr.op == Opcode::AllocTensor);
    size_t j = it->first;
    for (; j >= 0; j--) {
      if (instructions[j].op == Opcode::AllocStorage &&
          instructions[j].dst == instr.alloc_tensor.storage) {
        break;
      }
    }
    CHECK_LT(j, it->first);
    // Update the current alloc tensor instrution by replacing the current
    // storage index to the first-fit memory.
    if (instructions[it->first].alloc_tensor.storage != it->second) {
      Index last_use = UpdateRegisterUse(instructions, it->first,
                                         instructions[it->first].alloc_tensor.storage, it->second);
      UpdateLiveRange(it->second, last_use, live_range);
      // instructions[it->first].alloc_tensor.storage = it->second;
      // Mark the allocstorage instruction used by the alloctensor instruction to
      // be removed.
      redundant_alloc_storage.insert(j);
    }
  }

  std::vector<Instruction> ret;
  for (size_t i = 0; i < instructions.size(); i++) {
    if (redundant_alloc_storage.count(i) == 0) {
      ret.push_back(instructions[i]);
    }
  }
  return ret;
}

std::vector<Instruction> MemoryAlloc(const std::vector<Instruction>& instructions,
                                     LiveRangeMap& live_range) {
  auto ret = instructions;
  auto memory_candidates = MemoryCadidates(instructions);
  std::map<size_t, Index> remove_map;
  for (size_t i = 0; i < instructions.size(); i++) {
    const auto& instr = instructions[i];
    if (instr.op == Opcode::AllocTensor) {
      // Find the first-fit memory to reuse and update the live range.
      Index found = FindFirstFitMemory(live_range, memory_candidates, instr, i);
      if (found >= 0) {
        Instruction mem = memory_candidates[found].first;
        // Delete the current alloctensor instruction from memory candidates.
        for (size_t j = 0; j < memory_candidates.size(); j++) {
          if (memory_candidates[j].first.dst == instr.dst) {
            memory_candidates.erase(memory_candidates.begin() + j,
                                    memory_candidates.begin() + j + 1);
            break;
          }
        }
        // Record the last use of index of the register and extend the live
        // range for the found instruciton.
        Index last_use = UpdateRegisterUse(ret, i, instr.dst, mem.dst);
        UpdateLiveRange(mem.dst, last_use, live_range);
        // Mark the current alloctensor instruciton to be deleted.
        remove_map[i] = mem.alloc_tensor.storage;
      }
    }
  }

  ret = Cleanup(remove_map, ret, live_range);
  return ret;

}

runtime::vm::VMFunction Liveness(const runtime::vm::VMFunction& vm_func) {
  auto instructions = vm_func.instructions;
  auto live_range = LiveRange(instructions);
  
  // LOG(INFO) << "beforeeeeeee \n";
  // for (const auto& it : instructions) {
  //   std::cout << it << std::endl;
  // }

  instructions = MemoryAlloc(instructions, live_range);
  // LOG(INFO) << "afterrrrrr \n";
  // for (const auto& it : instructions) {
  //   std::cout << it << std::endl;
  // }
  instructions = UpdateRegisterId(instructions, vm_func.params.size());
  // instructions = ShuffleConstStorage(instructions);
  // instructions = UpdateRegisterId(instructions, vm_func.params.size());
  return runtime::vm::VMFunction(vm_func.name, vm_func.params, instructions,
                                 vm_func.register_file_size);
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
