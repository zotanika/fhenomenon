#pragma once

#include "Session/Session.h"

namespace fhenomenon {

template<typename Op>
void Session::run(Op&& ops) {
    LOG_MESSAGE("Session: begin");
    if (session_ptr_ == nullptr)
        session_ptr_ = shared_from_this();
    active_ = true;

    // Run lambda to capture operations
    ops();

    // Apply runtime optimization
    optimize();

    // Execute operations in optimized order
    active_ = false;
    //for (const auto& op : operations_) {
      //  LOG_MESSAGE("Session: execute operation");
    //    op->execute(); 
    //}
    LOG_MESSAGE("Session: end");
}

template <typename T> void Session::saveOp(std::shared_ptr<scheduler::Operation<T>> op) {
    LOG_MESSAGE("Session: operation added");
    operations_.push_back(std::move(op));
}

} // namespace fhenomenon
