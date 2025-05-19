#include "Session/Session.h"

using namespace fhenomenon;

thread_local std::shared_ptr<Session> Session::session_ptr_ = nullptr;

void Session::optimize() {
  LOG_MESSAGE("Session: optimize");
  std::vector<std::shared_ptr<fhenomenon::scheduler::OperationBase>> convertedOperations;
  switch (Backend::getInstance().getBackendType()) {
    case BackendType::BuiltinBackend:{
      LOG_MESSAGE("Session: optimize with builtin backend");
      auto &scheduler = Backend::getScheduler();
      scheduler.addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());
      //scheduler_.addStrategy(std::make_shared<scheduler::FuseOperationsStrategy>());
      //scheduler_.addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());
      //scheduler_.addStrategy(std::make_shared<scheduler::AdditionToMultiplicationStrategy>());
      //scheduler_.addStrategy(std::make_shared<scheduler::PrintOperationsStrategy>());
      scheduler::Planner<int> planner;
      planner = scheduler.buildGraph<int>(operations_, planner);
      scheduler.optimizeGraph(planner);
      scheduler.evaluateGraph(planner);
      
      //for (auto &operation : planner.getOperations()) {
        //  convertedOperations.push_back(std::make_shared<fhenomenon::scheduler::Operation<int>>(operation));
      //}
      break;
    }
    case BackendType::ExternalBackend:
      LOG_MESSAGE("Session: optimize with external backend");
      break;
  default:
    throw std::runtime_error("Unknown backend");
  }
}
