#pragma once

#include "FHN/Runtime.h"

#include <limits>

namespace fhenomenon::runtime {

// An opaque, scope-local symbol. It cannot reveal or convert to a concrete Value.
class ScopeValue {
  public:
  ScopeValue() noexcept = default;
  std::size_t id() const noexcept { return id_; }
  friend bool operator==(ScopeValue a, ScopeValue b) noexcept { return a.scope_ == b.scope_ && a.id_ == b.id_; }
  friend bool operator!=(ScopeValue a, ScopeValue b) noexcept { return !(a == b); }

  private:
  std::uint64_t scope_ = 0;
  std::size_t id_ = std::numeric_limits<std::size_t>::max();
  ScopeValue(std::uint64_t scope, std::size_t id) : scope_(scope), id_(id) {}
  friend class Scope;
  friend class ScopeView;
};
using ScopeValues = std::vector<ScopeValue>;

// Public provider metadata only. The core checks format/kind/bytes and leaves
// the contents and invariants of metadata to the provider contract callbacks.
struct ScopeDescriptor {
  std::string format;
  ResourceKind kind = ResourceKind::Value;
  std::size_t bytes = 0;
  std::any metadata;
};

struct ScopeContract {
  // All callbacks are required and must be metadata-only: Runtime evaluation,
  // allocation and recursive scope recording are forbidden during these calls.
  std::function<ScopeDescriptor(const Value &)> describe;
  std::function<std::vector<ScopeDescriptor>(const std::string &, const std::vector<ScopeDescriptor> &,
                                             const std::any &)>
    infer;
  std::function<void(const Value &, const ScopeDescriptor &)> validate;
};

// Limits apply independently to each runScope(), inspectScope() or native
// Session::run() call. Inspection uses the capture caps without invoking matchers.
// All structural caps must be positive, even when use_kernels is false.
// These bound recorded structure and matcher calls, not memory bytes or CPU time;
// use RuntimeOptions::max_live_bytes for charged ciphertext/artifact storage.
struct ScopeOptions {
  // Recorded submit() calls, including commands declaring zero outputs.
  std::size_t max_commands = 256;
  // Distinct imported identities + all declared output symbols, including dead ones.
  // Repeated input() calls for the same concrete Value count only once.
  std::size_t max_values = 2048;
  // Sum of command input counts; repeated uses of one symbol each count as an edge.
  std::size_t max_input_edges = 4096;
  // Entries returned by the capture callback; duplicate exports each count.
  std::size_t max_exports = 2048;
  // Maximum commands visible to one matcher, also limited by kernel.max_commands
  // and the remaining sequence. Shortens the view; it is not an overflow error.
  std::size_t max_lookahead = 32;
  // Total ScopeKernel::match calls over the whole run, including declines.
  // Needing another call after this limit raises BudgetExceeded, without fallback.
  std::size_t max_match_attempts = 4096;
  // False skips span matching but keeps capture, liveness and ordinary recipe
  // selection. Prepared/fused recipes and explicitly supplied state still work.
  bool use_kernels = true;
  // Optional strict metadata preflight. Each submit infers every output before
  // any execution starts. Outputs are checked again before publishing a span.
  // Recipe/kernel registration is frozen for the entire contracted scope.
  // Callback functions are copied at scope creation; captures must remain valid.
  std::shared_ptr<const ScopeContract> contract;
  // Wall-clock budget from capture start; zero is unlimited. Checked at capture
  // completion and span boundaries, not within a running provider operation.
  std::size_t max_duration_ms = 0;
};

struct ScopeCommand {
  std::string operation;
  ScopeValues inputs;
  // Supply an owning immutable snapshot; borrowed mutable storage is unsupported.
  std::any attributes;
  ScopeValues outputs;
};

class Scope;
// Views, subviews and returned references are borrowed only for the duration of
// a matcher, executor or inspection callback. Do not retain them after it returns.
// Inspection exposes concrete imported values; command outputs remain symbolic.
class ScopeView {
  public:
  std::size_t size() const noexcept;
  std::size_t start() const noexcept;
  // Bounds are relative to this view. Liveness still accounts for all uses and
  // exports in the original graph, including commands outside this subview.
  ScopeView subview(std::size_t localOffset, std::size_t length) const;
  const ScopeCommand &command(std::size_t localOffset) const;
  const Value *find(ScopeValue symbol) const;
  const Value &value(ScopeValue symbol) const;
  const ScopeDescriptor &descriptor(ScopeValue symbol) const;
  ScopeValues liveIns(std::size_t length) const;
  ScopeValues liveOuts(std::size_t length) const;

  private:
  const Scope *scope_;
  std::size_t start_;
  std::size_t length_;
  ScopeView(const Scope &scope, std::size_t start, std::size_t length);
  friend class Scope;
};

struct ScopeKernel {
  std::string name;
  int priority = 0;
  std::size_t max_commands = 1;
  std::function<std::size_t(const ScopeView &, std::string &reason)> match;
  // Return exactly view.liveOuts(view.size()), in that symbolic order.
  std::function<Values(Runtime &, const ScopeView &)> execute;
};

class Scope {
  public:
  ~Scope();
  bool belongsTo(const Runtime &runtime) const noexcept;
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  ScopeValue input(const Value &value);
  ScopeValues submit(std::string operation, ScopeValues inputs, std::any attributes = {}, std::size_t output_count = 1);
  // Requires a contract. The reference is valid until the next recording call
  // or scope destruction; descriptors do not expose concrete output values.
  const ScopeDescriptor &descriptor(ScopeValue symbol) const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Scope(Runtime &runtime, ScopeOptions options);
  Values execute(const ScopeValues &exports, const std::vector<std::shared_ptr<const ScopeKernel>> &kernels,
                 const std::function<void(TraceKind, const std::string &, const std::string &)> &record);
  void finishCapture(const ScopeValues &exports);
  void inspect(const ScopeValues &exports, const std::function<void(const ScopeView &)> &callback);
  void validate(ScopeValue symbol) const;
  friend class Runtime;
  friend class ScopeView;
};

} // namespace fhenomenon::runtime
