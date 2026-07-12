#include "corpus_oracle.h"

namespace fhenomenon {
namespace corpus {

namespace {

int64_t scalarOf(const FhnInstruction &inst) { return static_cast<int64_t>(inst.fparams[0]); }

} // namespace

std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &program, const std::map<uint32_t, Slots> &inputs) {
  std::map<uint32_t, Slots> vals = inputs;

  auto get = [&vals](uint32_t id) -> const Slots * {
    auto it = vals.find(id);
    return it == vals.end() ? nullptr : &it->second;
  };

  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    const Slots *a = get(inst.operands[0]);
    const Slots *b = get(inst.operands[1]);
    if (!a)
      return std::nullopt;

    auto binary = [&](auto fn) -> std::optional<Slots> {
      if (!b || b->size() != a->size())
        return std::nullopt;
      Slots out(a->size());
      for (size_t s = 0; s < out.size(); ++s)
        out[s] = fn((*a)[s], (*b)[s]);
      return out;
    };
    auto unary = [&](auto fn) -> Slots {
      Slots out(a->size());
      for (size_t s = 0; s < out.size(); ++s)
        out[s] = fn((*a)[s]);
      return out;
    };

    std::optional<Slots> out;
    switch (inst.opcode) {
    case FHN_ADD_CC:
      out = binary([](int64_t x, int64_t y) { return x + y; });
      break;
    case FHN_SUB_CC:
      out = binary([](int64_t x, int64_t y) { return x - y; });
      break;
    case FHN_MULT_CC:
      out = binary([](int64_t x, int64_t y) { return x * y; });
      break;
    case FHN_ADD_CS:
      out = unary([&inst](int64_t x) { return x + scalarOf(inst); });
      break;
    case FHN_MULT_CS:
      out = unary([&inst](int64_t x) { return x * scalarOf(inst); });
      break;
    case FHN_SUB_SC:
      out = unary([&inst](int64_t x) { return scalarOf(inst) - x; });
      break;
    case FHN_NEGATE:
      out = unary([](int64_t x) { return -x; });
      break;
    case FHN_ROTATE: {
      const int64_t n = static_cast<int64_t>(a->size());
      const int64_t d = ((inst.params[0] % n) + n) % n; // positive = left
      Slots rotated(a->size());
      for (int64_t s = 0; s < n; ++s)
        rotated[static_cast<size_t>(s)] = (*a)[static_cast<size_t>((s + d) % n)];
      out = rotated;
      break;
    }
    case FHN_EQ:
      out = binary([](int64_t x, int64_t y) { return x == y ? 1 : 0; });
      break;
    case FHN_LT:
      out = binary([](int64_t x, int64_t y) { return x < y ? 1 : 0; });
      break;
    case FHN_LE:
      out = binary([](int64_t x, int64_t y) { return x <= y ? 1 : 0; });
      break;
    case FHN_AND:
      out = binary([](int64_t x, int64_t y) { return x & y; });
      break;
    case FHN_OR:
      out = binary([](int64_t x, int64_t y) { return x | y; });
      break;
    case FHN_XOR:
      out = binary([](int64_t x, int64_t y) { return x ^ y; });
      break;
    default:
      return std::nullopt; // fused/lifecycle opcodes are not modeled
    }

    if (!out)
      return std::nullopt;
    vals[inst.result_id] = std::move(*out);
  }

  return vals;
}

} // namespace corpus
} // namespace fhenomenon
