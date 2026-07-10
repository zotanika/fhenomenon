#pragma once

#include "Backend/Backend.h"
#include "Fhenon.h"
#include "Parameter/CKKSParameter.h"
#include "Parameter/Parameter.h"
#include "Parameter/ParameterGen.h"
#include "Profile.h"
#include "Scheduler/ASTPass.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Scheduler.h"

#include <memory>
#include <unordered_map>

namespace fhenomenon {

class Session final : public std::enable_shared_from_this<Session> {
  public:
  static std::shared_ptr<Session> create(const Backend &backend) {
    session_ptr_ = std::shared_ptr<Session>(new Session(backend));
    return session_ptr_;
  }

  static std::shared_ptr<Session> getSession() { return session_ptr_; }

  ~Session() {
    if (session_ptr_.get() == this)
      session_ptr_ = nullptr;
  }

  template <typename T> void setEntity(const void *key, Fhenon<T> &entity) {
    entity_map_[key] = std::any(std::reference_wrapper<Fhenon<T>>(entity));
  }

  template <typename T> void saveEntity(Fhenon<T> &entity) {
    // Check if entity is already managed
    if (entity_map_.find(&entity) == entity_map_.end()) {
      entity_map_[&entity] = std::any(std::reference_wrapper<Fhenon<T>>(entity));
    }
  }

  template <typename T> void saveEntity(const void *key, Fhenon<T> &entity) {
    // Check if entity is already managed
    if (entity_map_.find(key) == entity_map_.end()) {
      entity_map_[key] = std::any(std::reference_wrapper<Fhenon<T>>(entity));
    }
  }

  template <typename T> Fhenon<T> *getEntity(Fhenon<T> &entity) {
    auto it = entity_map_.find(&entity);
    return it != entity_map_.end() ? &std::any_cast<std::reference_wrapper<Fhenon<T>>>(it->second).get() : nullptr;
  }

  template <typename T> Fhenon<T> *getEntity(const void *key) {
    auto it = entity_map_.find(key);
    return it != entity_map_.end() ? &std::any_cast<std::reference_wrapper<Fhenon<T>>>(it->second).get() : nullptr;
  }

  template <typename T> std::shared_ptr<Fhenon<T>> trackEntity(Fhenon<T> &entity) {
    // Resolve to the canonical object for this address: caller-owned
    // variables map to themselves, copies of operation results map to the
    // shared result they were copied from (registered by the copy
    // constructor).
    const void *key = &entity;
    Fhenon<T> *canonical = getEntity<T>(key);
    if (!canonical) {
      saveEntity(key, entity);
      canonical = &entity;
    }

    // Operation temporaries are shared-owned; share their ownership so they
    // outlive the recording expression.
    if (auto owned = canonical->weak_from_this().lock()) {
      LOG_MESSAGE("Tracking shared entity: " << owned.get() << ", " << owned->getValue());
      return owned;
    }

    // Caller-owned variables are aliased in place, never copied: every
    // recorded operation must observe earlier writes to the same variable
    // (read-after-write), and the final result must land in the caller's
    // object. The alias cannot outlive the variable because execution
    // completes inside run(), while the caller's scope is still alive.
    LOG_MESSAGE("Tracking caller entity: " << canonical << ", " << canonical->getValue());
    return std::shared_ptr<Fhenon<T>>(canonical, [](Fhenon<T> *) {});
  }

  template <typename T> void saveOp(std::shared_ptr<scheduler::Operation<T>> op);
  template <typename Op> void run(Op &&ops);

  bool isActive() { return (session_ptr_ == nullptr) ? false : active_; }

  void useBackend() const { (void)backend_; }

  private:
  // prevent direct instantiation

  explicit Session(const Backend &backend)
    : active_(false), backend_(backend), scheduler_(std::make_unique<scheduler::Scheduler>(backend)) {}

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  void optimize();

  // Drop all recording state so a later run() starts fresh: recorded
  // operations hold aliases to caller-owned variables, and entity_map_ holds
  // address-keyed entries that would otherwise resurrect dead expression
  // temporaries when the stack reuses their addresses.
  void endRun() {
    operations_.clear();
    entity_map_.clear();
    active_ = false;
  }

  bool active_;
  bool passes_registered_ = false;
  const Backend &backend_;
  std::unique_ptr<scheduler::Scheduler> scheduler_; // Scheduler with backend delegate
  // set of operations
  std::vector<std::shared_ptr<scheduler::OperationBase>> operations_;
  // Map for reference of Fhenon
  std::unordered_map<const void *, std::any> entity_map_;

  // `thread_local` variable to save current session
  static thread_local std::shared_ptr<Session> session_ptr_;
};

} // namespace fhenomenon

#include "Session/Session.tpp"
