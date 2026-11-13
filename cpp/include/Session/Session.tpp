#pragma once

#include "Session/Session.h"

#include <type_traits>
#include <utility>

namespace fhenomenon {

// Contract: Fhenon variables used inside `ops` must be declared in a scope
// enclosing this call (and belong() to a profile) — recorded operations alias
// them in place, and evaluation happens after the lambda body returns.
template <typename Op> auto Session::run(Op &&ops) {
  constexpr bool scoped = std::is_invocable_v<Op &, runtime::Scope &>;
  constexpr bool nullary = std::is_invocable_v<Op &>;
  static_assert(scoped != nullary, "Session::run needs either a nullary callback or a Scope& callback");
  if constexpr (scoped) {
    static_assert(std::is_same_v<std::invoke_result_t<Op &, runtime::Scope &>, runtime::ScopeValues>,
                  "Native scope callbacks must return ScopeValues to export");
    if (!native_runtime_)
      throw std::logic_error("Session: Scope& callback requires a native runtime session");
    if (isRecording())
      throw std::logic_error("Session: nested recording is not supported");
    auto previous = std::exchange(session_ptr_, shared_from_this());
    active_ = true;
    try {
      auto results = native_runtime_->runScope(
        [&](runtime::Scope &scope) {
          auto exports = ops(scope);
          if (poisoned_)
            throw std::logic_error("Session: " + poison_reason_);
          return exports;
        },
        scope_options_);
      // A contributed executor can also misuse the legacy frontend. Its
      // noexcept operations poison the Session; never publish a successful run.
      if (poisoned_)
        throw std::logic_error("Session: " + poison_reason_);
      endRun();
      session_ptr_ = std::move(previous);
      return results;
    } catch (...) {
      endRun();
      session_ptr_ = std::move(previous);
      throw;
    }
  } else {
    if (native_runtime_)
      throw std::logic_error("Session: native runtime session requires a Scope& callback");
    LOG_MESSAGE("Session: begin");
    if (isRecording())
      throw std::logic_error("Session: nested recording is not supported");
    auto previous = std::exchange(session_ptr_, shared_from_this());
    active_ = true;

    try {
      // Run lambda to capture operations
      ops();

      // Build, optimize, and evaluate the recorded graph.
      optimize();
    } catch (...) {
      endRun();
      session_ptr_ = std::move(previous);
      throw;
    }

    endRun();
    session_ptr_ = std::move(previous);
    LOG_MESSAGE("Session: end");
  }
}

template <typename T> void Session::saveOp(std::shared_ptr<scheduler::Operation<T>> op) {
  if (native_runtime_) {
    // Fhenon move assignment is noexcept. Poison capture rather than throwing
    // through that path; run() rejects it before any recorded native command executes.
    poisonRecording("use ScopeValue commands in a native scope, not Fhenon recording");
    // Keep operations solely for temporary ownership until endRun(). Returned
    // Fhenon shells reference these canonical objects even in a rejected capture.
  }
  LOG_MESSAGE("Session: operation added");
  if constexpr (!std::is_same_v<T, int>)
    unsupported_type_ = true;
  operations_.push_back(std::move(op));
}

} // namespace fhenomenon
