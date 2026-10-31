#pragma once

#include "Session/Session.h"

#include <type_traits>
#include <utility>

namespace fhenomenon {

// Contract: Fhenon variables used inside `ops` must be declared in a scope
// enclosing this call (and belong() to a profile) — recorded operations alias
// them in place, and evaluation happens after the lambda body returns.
template <typename Op> void Session::run(Op &&ops) {
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

template <typename T> void Session::saveOp(std::shared_ptr<scheduler::Operation<T>> op) {
  LOG_MESSAGE("Session: operation added");
  if constexpr (!std::is_same_v<T, int>)
    unsupported_type_ = true;
  operations_.push_back(std::move(op));
}

} // namespace fhenomenon
