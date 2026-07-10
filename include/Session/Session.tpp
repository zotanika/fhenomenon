#pragma once

#include "Session/Session.h"

namespace fhenomenon {

// Contract: Compuon variables used inside `ops` must be declared in a scope
// enclosing this call (and belong() to a profile) — recorded operations alias
// them in place, and evaluation happens after the lambda body returns.
template <typename Op> void Session::run(Op &&ops) {
  LOG_MESSAGE("Session: begin");
  if (session_ptr_ == nullptr)
    session_ptr_ = shared_from_this();
  active_ = true;

  try {
    // Run lambda to capture operations
    ops();

    // Build, optimize, and evaluate the recorded graph.
    optimize();
  } catch (...) {
    endRun();
    throw;
  }

  endRun();
  LOG_MESSAGE("Session: end");
}

template <typename T> void Session::saveOp(std::shared_ptr<scheduler::Operation<T>> op) {
  LOG_MESSAGE("Session: operation added");
  operations_.push_back(std::move(op));
}

} // namespace fhenomenon
