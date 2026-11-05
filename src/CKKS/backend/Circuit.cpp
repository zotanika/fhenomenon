#include "CKKS/Circuit.h"
#include "BackendInternal.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fhenomenon::ckks;

namespace {
constexpr uint32_t absent = UINT32_MAX;
constexpr size_t max_terms = 256;
struct Signature {
  CiphertextLayout layout;
  uint32_t slots;
};
bool sameScale(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(a, b); }
bool compatible(Signature a, Signature b) {
  return a.slots == b.slots && a.layout.poly == b.layout.poly && a.layout.num_polys == b.layout.num_polys &&
         sameScale(a.layout.scale, b.layout.scale);
}
int64_t checked(bool overflowed, int64_t value) {
  if (overflowed)
    throw std::invalid_argument("affine coefficient exceeds signed 64-bit range");
  return value;
}
int64_t checkedProduct(int64_t a, int64_t b) {
  int64_t out = 0;
  return checked(__builtin_mul_overflow(a, b, &out), out);
}
// sign is +1 or -1; subtracting directly keeps INT64_MIN coefficients exact.
int64_t checkedSum(int64_t a, int64_t b, int64_t sign) {
  int64_t out = 0;
  return checked(sign < 0 ? __builtin_sub_overflow(a, b, &out) : __builtin_add_overflow(a, b, &out), out);
}
bool integer(double value, int64_t &out) {
  if (!std::isfinite(value) || std::trunc(value) != value)
    return false;
  if (value < -9223372036854775808.0 || value >= 9223372036854775808.0)
    throw std::invalid_argument("integer scalar exceeds signed 64-bit range");
  out = static_cast<int64_t>(value);
  return true;
}
struct Form {
  std::map<uint32_t, int64_t> terms;
  double constant = 0;
  uint32_t anchor = 0;
};
Form scaled(Form form, int64_t coefficient) {
  for (auto &term : form.terms)
    term.second = checkedProduct(term.second, coefficient);
  if (!coefficient)
    form.terms.clear();
  form.constant *= static_cast<double>(coefficient);
  if (!std::isfinite(form.constant))
    throw std::invalid_argument("affine constant overflow");
  return form;
}
Form sum(Form a, const Form &b, int64_t sign) {
  for (auto [id, coefficient] : b.terms) {
    auto &target = a.terms[id];
    target = checkedSum(target, coefficient, sign);
    if (!target)
      a.terms.erase(id);
  }
  a.constant += b.constant * static_cast<double>(sign);
  if (!std::isfinite(a.constant) || a.terms.size() > max_terms)
    throw std::invalid_argument("affine expression exceeds bounded form capacity");
  return a;
}
struct Source {
  FhnCircuitNode node{};
  uint32_t args[2]{absent, absent};
  uint32_t uses = 0;
  Form form;
  uint32_t materialized = absent;
  Signature nominal{};
};
struct Value {
  Signature signature;
  uint32_t slot = absent;
};
struct KeyBinding {
  SwitchKeyKind kind;
  uint64_t galois;
  std::weak_ptr<const EvaluationKey> weak;
  std::shared_ptr<const EvaluationKey> lease;
};
enum class Kind { Linear, Product, ProductSum, Rotation, Fractional, LevelDown };
struct Operation {
  Kind kind = Kind::Linear;
  uint32_t output = absent;
  std::vector<uint32_t> inputs;
  std::vector<int64_t> coefficients;
  std::vector<region::LinearTerm> linear;
  std::vector<region::ProductTerm> products;
  uint32_t product_count = 0;
  uint32_t key = absent;
  uint64_t galois = 1;
  double scalar = 0;
  uint32_t target_level = 0;
};
} // namespace

struct FhnExecutable {
  FhnBackendCtx *context = nullptr;
  std::vector<Source> sources;
  std::vector<Signature> input_signatures;
  std::vector<uint32_t> output_ids, outputs;
  std::vector<Value> values;
  std::vector<Operation> operations;
  std::vector<KeyBinding> keys;
  std::vector<PolySlab> slabs;
  std::vector<ConstCiphertextView> views;
  Arena scratch{0};
  FhnExecutableStats stats{};
};

namespace {
class Compiler {
  public:
  FhnExecutable &e;
  const EvaluationTables &tables;
  size_t depth = 0;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> lowered_levels;
  explicit Compiler(FhnExecutable &value) : e(value), tables(*value.context->tables) {}
  uint32_t key(SwitchKeyKind kind, uint64_t galois) {
    for (uint32_t i = 0; i < e.keys.size(); ++i)
      if (e.keys[i].kind == kind && e.keys[i].galois == galois)
        return i;
    e.keys.push_back({kind, galois, {}, {}});
    return static_cast<uint32_t>(e.keys.size() - 1);
  }
  uint32_t emit(Operation operation, Signature signature) {
    if (!signature.layout.wellFormed() || signature.layout.num_polys != 2)
      throw std::invalid_argument("invalid inferred ciphertext signature");
    operation.output = static_cast<uint32_t>(e.values.size());
    e.values.push_back({signature, absent});
    if (operation.kind == Kind::Linear || operation.kind == Kind::Rotation || operation.kind == Kind::ProductSum)
      operation.linear.resize(operation.coefficients.size() - operation.product_count);
    operation.products.resize(operation.product_count);
    e.operations.push_back(std::move(operation));
    return static_cast<uint32_t>(e.values.size() - 1);
  }
  uint32_t levelDown(uint32_t value, uint32_t q_count) {
    auto signature = e.values[value].signature;
    if (signature.layout.poly.basis.main_end == q_count)
      return value;
    const auto id = std::make_pair(value, q_count);
    auto found = lowered_levels.find(id);
    if (found != lowered_levels.end())
      return found->second;
    if (!q_count || q_count > signature.layout.poly.basis.main_end)
      throw std::invalid_argument("invalid level alignment");
    signature.layout.poly.basis.main_end = q_count;
    Operation op;
    op.kind = Kind::LevelDown;
    op.inputs = {value};
    op.target_level = q_count - 1;
    const auto result = emit(std::move(op), signature);
    lowered_levels.emplace(id, result);
    return result;
  }
  Signature align(std::vector<uint32_t> &values, bool scales) {
    auto target = e.values[values.front()].signature;
    uint32_t q_count = target.layout.poly.basis.main_end;
    for (auto id : values) {
      const auto signature = e.values[id].signature;
      if (signature.slots != target.slots || !signature.layout.poly.sameRing(target.layout.poly) ||
          signature.layout.num_polys != 2 || (scales && !sameScale(signature.layout.scale, target.layout.scale)))
        throw std::invalid_argument("incompatible packing or actual scales at circuit join");
      q_count = std::min(q_count, signature.layout.poly.basis.main_end);
    }
    for (auto &id : values)
      id = levelDown(id, q_count);
    target.layout.poly.basis.main_end = q_count;
    return target;
  }
  Signature productOutput(Signature a, double second_scale) {
    if (a.layout.poly.basis.main_end < 2)
      throw std::invalid_argument("circuit multiplication exhausts modulus chain");
    --a.layout.poly.basis.main_end;
    a.layout.scale =
      (a.layout.scale * second_scale) / static_cast<double>(tables.params().mainPrimes()[a.layout.poly.basis.main_end]);
    if (!std::isfinite(a.layout.scale) || a.layout.scale <= 0)
      throw std::invalid_argument("product scale overflow");
    return a;
  }
  void checkConstant(double constant, double scale) {
    int64_t encoded = 0;
    if (!region::encodeConstant(constant, scale, encoded))
      throw std::invalid_argument("constant cannot be encoded at inferred scale");
  }
  uint32_t materialize(uint32_t id) {
    if (e.sources[id].materialized != absent)
      return e.sources[id].materialized;
    if (++depth > 512)
      throw std::invalid_argument("circuit nesting exceeds compiler limit");
    const Source source = e.sources[id];
    uint32_t result = absent;
    const bool self = source.form.terms.size() == 1 && source.form.terms.begin()->first == id &&
                      source.form.terms.begin()->second == 1 && source.form.constant == 0;
    if (!self) {
      result = affine(source.form);
    } else {
      Operation op;
      op.inputs = {materialize(source.args[0])};
      auto signature = e.values[op.inputs[0]].signature;
      switch (source.node.op) {
      case FHN_CIRCUIT_MULTIPLY:
        op.inputs.push_back(materialize(source.args[1]));
        signature = align(op.inputs, false);
        signature = productOutput(signature, e.values[op.inputs[1]].signature.layout.scale);
        op.kind = Kind::Product;
        op.key = key(SwitchKeyKind::Relinearization, 1);
        break;
      case FHN_CIRCUIT_ROTATE:
      case FHN_CIRCUIT_CONJUGATE:
        op.kind = Kind::Rotation;
        op.galois = source.node.op == FHN_CIRCUIT_CONJUGATE
                      ? 2 * tables.params().degree() - 1
                      : eval::galoisElement(source.node.rotation, tables.params().logDegree());
        if (op.galois != 1)
          op.key = key(SwitchKeyKind::Galois, op.galois);
        break;
      case FHN_CIRCUIT_SCALE:
        op.kind = Kind::Fractional;
        op.scalar = source.node.scalar;
        if (signature.layout.poly.basis.main_end < 2)
          throw std::invalid_argument("fractional scaling exhausts modulus chain");
        checkConstant(op.scalar,
                      static_cast<double>(tables.params().mainPrimes()[signature.layout.poly.basis.main_end - 1]));
        --signature.layout.poly.basis.main_end;
        break;
      default:
        throw std::invalid_argument("unexpected semantic circuit leaf");
      }
      result = emit(std::move(op), signature);
    }
    e.sources[id].materialized = result;
    --depth;
    return result;
  }
  uint32_t affine(const Form &form) {
    Operation quadratic;
    quadratic.kind = Kind::ProductSum;
    quadratic.scalar = form.constant;
    std::vector<std::pair<uint32_t, int64_t>> remainder;
    // Examine quadratic leaves before materializing them. Linear operands
    // remain actual typed values, so a failed rewrite never relabels a scale.
    for (auto [id, coefficient] : form.terms) {
      const auto &source = e.sources[id];
      if (id >= e.input_signatures.size() && source.node.op == FHN_CIRCUIT_MULTIPLY && source.uses == 1 &&
          source.materialized == absent) {
        const auto left = source.args[0], right = source.args[1];
        quadratic.inputs.push_back(materialize(left));
        quadratic.inputs.push_back(materialize(right));
        quadratic.coefficients.push_back(coefficient);
        ++quadratic.product_count;
      } else {
        remainder.emplace_back(id, coefficient);
      }
    }
    if (quadratic.product_count) {
      for (auto [id, coefficient] : remainder) {
        quadratic.inputs.push_back(materialize(id));
        quadratic.coefficients.push_back(coefficient);
      }
      const auto common = e.values[quadratic.inputs.front()].signature;
      bool valid = common.layout.poly.basis.main_end >= 2;
      for (auto id : quadratic.inputs)
        valid = valid && compatible(common, e.values[id].signature);
      int64_t common_integer = 0;
      if (!remainder.empty() || form.constant != 0)
        valid = valid && integer(common.layout.scale, common_integer) && common_integer > 0;
      if (valid) {
        checkConstant(form.constant, common.layout.scale);
        quadratic.key = key(SwitchKeyKind::Relinearization, 1);
        return emit(std::move(quadratic), productOutput(common, common.layout.scale));
      }
    }
    uint32_t rotation_id = absent;
    if (form.constant == 0)
      for (auto [id, coefficient] : form.terms) {
        const auto &source = e.sources[id];
        if (id >= e.input_signatures.size() && source.materialized == absent && source.uses == 1 && coefficient == 1 &&
            (source.node.op == FHN_CIRCUIT_ROTATE || source.node.op == FHN_CIRCUIT_CONJUGATE)) {
          rotation_id = id;
          break;
        }
      }
    Operation op;
    op.kind = rotation_id == absent ? Kind::Linear : Kind::Rotation;
    op.scalar = form.constant;
    if (rotation_id != absent) {
      const auto &source = e.sources[rotation_id];
      op.inputs.push_back(materialize(source.args[0]));
      op.galois = source.node.op == FHN_CIRCUIT_CONJUGATE
                    ? 2 * tables.params().degree() - 1
                    : eval::galoisElement(source.node.rotation, tables.params().logDegree());
      if (op.galois != 1)
        op.key = key(SwitchKeyKind::Galois, op.galois);
    }
    for (auto [id, coefficient] : form.terms)
      if (id != rotation_id) {
        op.inputs.push_back(materialize(id));
        op.coefficients.push_back(coefficient);
      }
    auto signature = op.inputs.empty() ? e.sources[form.anchor].nominal : align(op.inputs, true);
    checkConstant(form.constant, signature.layout.scale);
    return emit(std::move(op), signature);
  }
  void import(const FhnCircuit &c, const FhnBuffer *const *inputs) {
    std::unordered_map<uint32_t, uint32_t> ids;
    for (uint32_t i = 0; i < c.num_inputs; ++i) {
      const auto *input = inputs[i];
      if (!c.input_ids[i] || !ids.emplace(c.input_ids[i], i).second || !input || input->owner != e.context ||
          !input->certified || !input->logical_slots || input->layout.num_polys != 2 ||
          !region::validateInput(input->view(), tables))
        throw std::invalid_argument("invalid circuit prototype input or input id");
      Signature signature{input->layout, input->logical_slots};
      e.input_signatures.push_back(signature);
      e.values.push_back({signature, absent});
      Source source;
      source.form = {{{i, 1}}, 0, i};
      source.materialized = i;
      source.nominal = signature;
      e.sources.push_back(std::move(source));
    }
    for (uint32_t i = 0; i < c.num_nodes; ++i) {
      Source source;
      source.node = c.nodes[i];
      const auto &node = source.node;
      const auto id = static_cast<uint32_t>(e.sources.size());
      if (!node.result_id || ids.count(node.result_id) || node.op > FHN_CIRCUIT_CONJUGATE)
        throw std::invalid_argument("duplicate result id or unknown circuit operation");
      const bool binary =
        node.op == FHN_CIRCUIT_ADD || node.op == FHN_CIRCUIT_SUBTRACT || node.op == FHN_CIRCUIT_MULTIPLY;
      if (!node.operands[0] || binary != (node.operands[1] != 0))
        throw std::invalid_argument("invalid circuit operation arity");
      for (uint32_t j = 0; j < (binary ? 2u : 1u); ++j) {
        const auto found = ids.find(node.operands[j]);
        if (found == ids.end())
          throw std::invalid_argument("circuit operand used before its definition");
        source.args[j] = found->second;
        ++e.sources[found->second].uses;
      }
      if ((node.op == FHN_CIRCUIT_SCALE || node.op == FHN_CIRCUIT_ADD_CONSTANT) && !std::isfinite(node.scalar))
        throw std::invalid_argument("nonfinite circuit scalar");
      const auto &a = e.sources[source.args[0]];
      if (binary && a.nominal.slots != e.sources[source.args[1]].nominal.slots)
        throw std::invalid_argument("incompatible packing at circuit operation");
      source.nominal = a.nominal;
      source.form = {{{id, 1}}, 0, id};
      int64_t coefficient = 0;
      switch (node.op) {
      case FHN_CIRCUIT_ADD:
      case FHN_CIRCUIT_SUBTRACT:
        source.form = sum(a.form, e.sources[source.args[1]].form, node.op == FHN_CIRCUIT_ADD ? 1 : -1);
        break;
      case FHN_CIRCUIT_NEGATE:
        source.form = scaled(a.form, -1);
        break;
      case FHN_CIRCUIT_SCALE:
        if (integer(node.scalar, coefficient))
          source.form = scaled(a.form, coefficient);
        else if (source.nominal.layout.poly.basis.main_end > 1)
          --source.nominal.layout.poly.basis.main_end;
        break;
      case FHN_CIRCUIT_ADD_CONSTANT:
        source.form = a.form;
        source.form.constant += node.scalar;
        if (!std::isfinite(source.form.constant))
          throw std::invalid_argument("affine constant overflow");
        break;
      case FHN_CIRCUIT_MULTIPLY:
        // Nominal types are used only for fully cancelled expressions. Real
        // scale joins and chain exhaustion are checked after region rewrites.
        if (source.nominal.layout.poly.basis.main_end > 1)
          source.nominal = productOutput(source.nominal, e.sources[source.args[1]].nominal.layout.scale);
        break;
      default:
        break;
      }
      e.sources.push_back(std::move(source));
      ids.emplace(node.result_id, id);
    }
    for (uint32_t i = 0; i < c.num_outputs; ++i) {
      const auto found = ids.find(c.output_ids[i]);
      if (found == ids.end())
        throw std::invalid_argument("circuit output is undefined");
      ++e.sources[found->second].uses;
      e.output_ids.push_back(c.output_ids[i]);
    }
    // Rebuild forms with complete use counts. Keep shared affine values that
    // contain expensive leaves as materialization boundaries; expanding them
    // into every consumer would duplicate rotations/products. Pure affine
    // input expressions remain compact and can safely be shared algebraically.
    auto childForm = [&](uint32_t id) {
      const auto &source = e.sources[id];
      if (source.uses > 1 && id >= e.input_signatures.size()) {
        for (const auto &term : source.form.terms)
          if (term.first >= e.input_signatures.size())
            return Form{{{id, 1}}, 0, id};
      }
      return source.form;
    };
    for (uint32_t i = static_cast<uint32_t>(e.input_signatures.size()); i < e.sources.size(); ++i) {
      auto &source = e.sources[i];
      int64_t coefficient = 0;
      switch (source.node.op) {
      case FHN_CIRCUIT_ADD:
      case FHN_CIRCUIT_SUBTRACT:
        source.form =
          sum(childForm(source.args[0]), childForm(source.args[1]), source.node.op == FHN_CIRCUIT_ADD ? 1 : -1);
        break;
      case FHN_CIRCUIT_NEGATE:
        source.form = scaled(childForm(source.args[0]), -1);
        break;
      case FHN_CIRCUIT_SCALE:
        if (integer(source.node.scalar, coefficient))
          source.form = scaled(childForm(source.args[0]), coefficient);
        break;
      case FHN_CIRCUIT_ADD_CONSTANT:
        source.form = childForm(source.args[0]);
        source.form.constant += source.node.scalar;
        break;
      default:
        break;
      }
    }
    for (auto id : e.output_ids) {
      auto value = materialize(ids.at(id));
      if (value < e.input_signatures.size()) {
        Operation copy;
        copy.inputs = {value};
        copy.coefficients = {1};
        value = emit(std::move(copy), e.values[value].signature);
      }
      e.outputs.push_back(value);
    }
  }
  void allocate(uint64_t limit) {
    std::vector<uint32_t> remaining(e.values.size(), 0);
    for (const auto &op : e.operations)
      for (auto id : op.inputs)
        ++remaining[id];
    for (auto id : e.outputs)
      ++remaining[id];
    std::vector<size_t> sizes;
    std::vector<bool> free;
    for (const auto &op : e.operations) {
      size_t slot = 0;
      while (slot < free.size() && !free[slot])
        ++slot;
      const auto words = e.values[op.output].signature.layout.wordCount();
      if (slot == free.size()) {
        sizes.push_back(words);
        free.push_back(false);
      } else {
        sizes[slot] = std::max(sizes[slot], words);
        free[slot] = false;
      }
      e.values[op.output].slot = static_cast<uint32_t>(slot);
      for (auto id : op.inputs)
        if (!--remaining[id] && e.values[id].slot != absent)
          free[e.values[id].slot] = true;
      if (!remaining[op.output])
        free[slot] = true;
    }
    e.stats = {};
    e.stats.struct_size = sizeof(e.stats);
    e.stats.source_nodes = static_cast<uint32_t>(e.sources.size() - e.input_signatures.size());
    e.stats.regions = static_cast<uint32_t>(e.operations.size());
    e.stats.storage_slots = static_cast<uint32_t>(sizes.size());
    const auto scratch_request = region::scratchBytes(tables.params());
    if (scratch_request > SIZE_MAX - (Arena::kAlignment - 1))
      throw std::invalid_argument("scratch reserve alignment overflows size");
    e.stats.scratch_bytes = ((scratch_request + Arena::kAlignment - 1) / Arena::kAlignment) * Arena::kAlignment;
    for (auto words : sizes) {
      if (words > (UINT64_MAX - e.stats.storage_bytes) / sizeof(uint64_t))
        throw std::invalid_argument("internal storage size overflow");
      e.stats.storage_bytes += words * sizeof(uint64_t);
    }
    if (e.stats.storage_bytes > UINT64_MAX - e.stats.scratch_bytes ||
        (limit && e.stats.storage_bytes + e.stats.scratch_bytes > limit))
      throw std::invalid_argument("circuit workspace exceeds requested byte limit");
    for (const auto &op : e.operations) {
      e.stats.linear_regions += op.kind == Kind::Linear;
      e.stats.product_regions += op.kind == Kind::Product || op.kind == Kind::ProductSum;
      e.stats.rotation_regions += op.kind == Kind::Rotation;
    }
    for (auto words : sizes)
      e.slabs.emplace_back(words);
    e.scratch = Arena(static_cast<size_t>(e.stats.scratch_bytes));
    e.views.resize(e.values.size());
  }
};
void diagnostic(char *error, uint32_t capacity, const char *message) noexcept {
  if (error && capacity) {
    const size_t count = std::min(static_cast<size_t>(capacity - 1), std::strlen(message));
    std::memcpy(error, message, count);
    error[count] = 0;
  }
}
void publish(FhnBuffer *output, ConstCiphertextView value, uint32_t slots) {
  for (uint32_t poly = 0; poly < value.layout.num_polys; ++poly)
    std::memcpy(output->slab.data() + poly * output->owner->stride, value.data + poly * value.poly_stride_words,
                value.layout.poly.wordCount() * sizeof(uint64_t));
  output->layout = value.layout;
  output->logical_slots = slots;
  output->certified = true;
}
} // namespace

extern "C" {
FhnExecutable *ckks_fhn_compile_circuit(FhnBackendCtx *ctx, const FhnCircuit *c, const FhnBuffer *const *inputs,
                                        uint32_t count, const FhnCompileOptions *options, char *error,
                                        uint32_t capacity) {
  diagnostic(error, capacity, "");
  try {
    if (!ctx || !c || c->version != FHN_CIRCUIT_VERSION || !c->num_inputs || !c->num_outputs ||
        count != c->num_inputs || !inputs || !c->input_ids || !c->output_ids || (c->num_nodes && !c->nodes) ||
        uint64_t{c->num_nodes} + c->num_inputs > 65536 || c->num_outputs > 65536 ||
        (options && options->struct_size != sizeof(FhnCompileOptions)))
      throw std::invalid_argument("malformed circuit or compile options");
    auto executable = std::make_unique<FhnExecutable>();
    executable->context = ctx;
    Compiler compiler(*executable);
    compiler.import(*c, inputs);
    compiler.allocate(options ? options->workspace_limit_bytes : 0);
    return executable.release();
  } catch (const std::exception &failure) {
    diagnostic(error, capacity, failure.what());
    return nullptr;
  } catch (...) {
    diagnostic(error, capacity, "circuit compilation failed");
    return nullptr;
  }
}
int ckks_fhn_client_prepare_executable(FhnClientCtx *client, FhnExecutable *e) {
  if (!client || !e || client->backend != e->context)
    return -1;
  try {
    auto *ctx = e->context;
    for (auto &binding : e->keys) {
      std::shared_ptr<const EvaluationKey> owned;
      if (binding.kind == SwitchKeyKind::Relinearization) {
        if (!client->relin)
          client->relin = generateRelinearizationKey(*client->secret, *ctx->tables, client->random, client->scratch);
        owned = client->relin;
        ctx->relin = owned;
      } else {
        auto found = client->rotations.find(binding.galois);
        if (found == client->rotations.end()) {
          auto generated =
            generateGaloisKey(*client->secret, binding.galois, *ctx->tables, client->random, client->scratch);
          if (!generated)
            return -1;
          found = client->rotations.emplace(binding.galois, std::move(generated)).first;
        }
        owned = found->second;
        ctx->rotations.insert_or_assign(binding.galois, owned);
      }
      if (!owned || !region::validateKey(owned->view(), binding.kind, binding.galois, *ctx->tables))
        return -1;
      binding.weak = owned;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}
int ckks_fhn_run_executable(FhnExecutable *e, const FhnBuffer *const *inputs, uint32_t input_count,
                            FhnBuffer *const *outputs, uint32_t output_count) {
  if (!e || !inputs || !outputs || input_count != e->input_signatures.size() || output_count != e->outputs.size())
    return -1;
  struct LeaseReset {
    FhnExecutable &e;
    ~LeaseReset() {
      for (auto &key : e.keys)
        key.lease.reset();
    }
  } reset{*e};
  try {
    for (uint32_t i = 0; i < input_count; ++i) {
      const auto *input = inputs[i];
      if (!input || input->owner != e->context || !input->certified || !input->view().wellFormed() ||
          !compatible(e->input_signatures[i], {input->layout, input->logical_slots}))
        return -1;
      e->views[i] = input->view();
    }
    for (uint32_t i = 0; i < output_count; ++i) {
      if (!outputs[i] || outputs[i]->owner != e->context)
        return -1;
      for (uint32_t j = 0; j < i; ++j)
        if (outputs[j] == outputs[i] && e->output_ids[j] != e->output_ids[i])
          return -1;
    }
    for (auto &key : e->keys) {
      key.lease = key.weak.lock();
      if (!key.lease)
        return -1;
    }
    const auto &tables = *e->context->tables;
    e->scratch.reset();
    for (auto &op : e->operations) {
      const auto &value = e->values[op.output];
      auto &slab = e->slabs[value.slot];
      CiphertextView out{slab.data(), slab.capacityWords(), value.signature.layout.poly.wordCount(),
                         value.signature.layout};
      const auto key = op.key == absent ? EvaluationKeyView{} : e->keys[op.key].lease->view();
      const size_t linear_offset = op.kind == Kind::Rotation ? 1 : 2 * op.product_count;
      for (size_t i = 0; i < op.linear.size(); ++i)
        op.linear[i] = {e->views[op.inputs[linear_offset + i]], op.coefficients[op.product_count + i]};
      for (uint32_t i = 0; i < op.product_count; ++i)
        op.products[i] = {e->views[op.inputs[2 * i]], e->views[op.inputs[2 * i + 1]], op.coefficients[i]};
      bool ok = false;
      switch (op.kind) {
      case Kind::Linear:
        ok = region::linear(out, op.linear.data(), op.linear.size(), op.scalar, value.signature.layout, tables,
                            e->scratch);
        break;
      case Kind::Product:
        ok = region::multiply(out, e->views[op.inputs[0]], e->views[op.inputs[1]], key, tables, e->scratch);
        break;
      case Kind::ProductSum:
        ok = region::productSum(out, op.products.data(), op.products.size(), op.linear.data(), op.linear.size(),
                                op.scalar, key, tables, e->scratch);
        break;
      case Kind::Rotation:
        ok = region::rotateAdd(out, e->views[op.inputs[0]], op.galois, key, op.linear.data(), op.linear.size(), tables,
                               e->scratch);
        break;
      case Kind::Fractional:
        ok = region::scaleFractional(out, e->views[op.inputs[0]], op.scalar, tables, e->scratch);
        break;
      case Kind::LevelDown:
        ok = region::levelDown(out, e->views[op.inputs[0]], op.target_level, tables, e->scratch);
        break;
      }
      if (!ok || !compatible(value.signature, {out.layout, value.signature.slots}))
        return -1;
      e->views[op.output] = asConst(out);
    }
    for (uint32_t i = 0; i < output_count; ++i)
      publish(outputs[i], e->views[e->outputs[i]], e->values[e->outputs[i]].signature.slots);
    return 0;
  } catch (...) {
    return -1;
  }
}
int ckks_fhn_executable_stats(const FhnExecutable *e, FhnExecutableStats *out) {
  if (!e || !out || out->struct_size != sizeof(FhnExecutableStats))
    return -1;
  *out = e->stats;
  return 0;
}
void ckks_fhn_executable_free(FhnExecutable *e) { delete e; }
}
