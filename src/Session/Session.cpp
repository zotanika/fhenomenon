#include "Session/Session.h"
#include "Scheduler/MatMulRecognitionPass.h"

using namespace fhenomenon;

thread_local std::shared_ptr<Session> Session::session_ptr_ = nullptr;

void Session::optimize() {
  LOG_MESSAGE("Session: optimize");

  if (!scheduler_) {
    throw std::runtime_error("Session: Scheduler not initialized");
  }

  // Register pre-AST passes
  scheduler_->addPreASTPass(std::make_shared<scheduler::MatMulRecognitionPass>());

  // Scheduler uses backend as delegate - it's decoupled from actual computation
  scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::FuseOperationsASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::AdditionToMultiplicationASTPass>());
  // scheduler_->addASTPass(std::make_shared<scheduler::PrintASTPass>());

  scheduler::Planner<int> planner;
  planner = scheduler_->buildGraph<int>(operations_, planner);
  scheduler_->optimizeGraph(planner);
  scheduler_->evaluateGraph(planner);
}
