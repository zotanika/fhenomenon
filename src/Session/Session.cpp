#include "Session/Session.h"

using namespace fhenomenon;

thread_local std::shared_ptr<Session> Session::session_ptr_ = nullptr;

void Session::optimize() {
  LOG_MESSAGE("Session: optimize");

  if (!scheduler_) {
    throw std::runtime_error("Session: Scheduler not initialized");
  }

  // Scheduler uses backend as delegate - it's decoupled from actual computation
  scheduler_->addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());
  // scheduler_->addStrategy(std::make_shared<scheduler::FuseOperationsStrategy>());
  // scheduler_->addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());
  // scheduler_->addStrategy(std::make_shared<scheduler::AdditionToMultiplicationStrategy>());
  // scheduler_->addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());

  scheduler::Planner<int> planner;
  planner = scheduler_->buildGraph<int>(operations_, planner);
  scheduler_->optimizeGraph(planner);
  scheduler_->evaluateGraph(planner);
}
