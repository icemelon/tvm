#include <tvm/runtime/vm.h>

#include <map>

namespace tvm {
namespace relay {
namespace vm {

struct Interval;
using Index = int64_t;
using LiveRangeMap = std::map<Index, Interval>;

struct Interval {
  Interval() {}
  Interval(Index start, Index end) : start(start), end(end) {}
  Index start{std::numeric_limits<Index>::max()};
  Index end{0};
};

LiveRangeMap LiveRange(const std::vector<runtime::vm::Instruction>& instructions);
runtime::vm::VMFunction Liveness(const runtime::vm::VMFunction& vm_func);

}  // namespace vm
}  // namespace relay
}  // namespace tvm
