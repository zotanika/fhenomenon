#pragma once

#include "Backend/Backend.h"
#include "Compuon.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Scheduler.h"
#include "Scheduler/Strategy.h"
#include "Parameter/Parameter.h"
#include "Parameter/ParameterGen.h"
#include "Parameter/CKKSParameter.h"
#include "Profile.h"

#include <unordered_map>
#include <memory>

namespace fhenomenon {

class Session final : public std::enable_shared_from_this<Session> {
  public:
  static std::shared_ptr<Session> create(const Backend &backend) {
    session_ptr_ = std::shared_ptr<Session>(new Session(backend));
    return session_ptr_;
  }

  static std::shared_ptr<Session> getSession() { return session_ptr_; }

  ~Session() { session_ptr_ = nullptr; }

  template <typename T> void setEntity(const void *key, Compuon<T> &entity){
    entity_map_[key] = std::any(std::reference_wrapper<Compuon<T>>(entity));
  }

  template <typename T> void saveEntity(Compuon<T> &entity) {
    // Check if entity is already managed
    if (entity_map_.find(&entity) == entity_map_.end()) {
      entity_map_[&entity] = std::any(std::reference_wrapper<Compuon<T>>(entity));
    }
  }

  template <typename T> void saveEntity(const void *key, Compuon<T> &entity) {
    // Check if entity is already managed
    if (entity_map_.find(key) == entity_map_.end()) {
      entity_map_[key] = std::any(std::reference_wrapper<Compuon<T>>(entity));
    }
  }

  template <typename T> Compuon<T> *getEntity(Compuon<T> &entity) {
    auto it = entity_map_.find(&entity);
    return it != entity_map_.end() ? &std::any_cast<std::reference_wrapper<Compuon<T>>>(it->second).get() : nullptr;
  }

  template <typename T> Compuon<T> *getEntity(const void *key) {
    auto it = entity_map_.find(key);
    return it != entity_map_.end() ? &std::any_cast<std::reference_wrapper<Compuon<T>>>(it->second).get() : nullptr;
  }

  template <typename T> std::shared_ptr<Compuon<T>> trackEntity(Compuon<T> &entity) {
    const void* key = &entity;
    auto existing_entity = getEntity<T>(key);
    if(existing_entity != nullptr){
      LOG_MESSAGE("Existing");
    }
    auto entity_p = std::shared_ptr<Compuon<T>>(existing_entity, [](Compuon<T>*){}); 
    //double delete bug if std::shared_ptr<Compuon<T>>(existing_entity);
    auto entity_ptr = entity_p ? entity_p->weak_from_this().lock() : entity.weak_from_this().lock();

    if (!entity_ptr) {
      LOG_MESSAGE("No Ptr");
      entity_ptr = std::make_shared<Compuon<T>>(entity);
      tmp_entities_.emplace_back(entity_ptr);
      saveEntity(&entity, *entity_ptr);
    }

    LOG_MESSAGE("Tracking entity: " << entity_ptr.get() << ", " << entity_ptr->getValue());

    return entity_ptr;
  }

  template <typename T> void saveOp(std::shared_ptr<scheduler::Operation<T>> op);
  template <typename Op> void run(Op &&ops);

  bool isActive() { return (session_ptr_ == nullptr) ? false : active_; }

  void useBackend() const {
        (void)backend_;
    }

  private:
  // prevent direct instantiation

  explicit Session(const Backend &backend) 
    : active_(false), backend_(backend), scheduler_(std::make_unique<scheduler::Scheduler>(backend)) {}

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  
  void optimize();
  
  bool active_;
  const Backend &backend_;
  std::unique_ptr<scheduler::Scheduler> scheduler_;  // Scheduler with backend delegate
  // set of operations
  std::vector<std::shared_ptr<scheduler::OperationBase>> operations_;
  std::vector<std::shared_ptr<void>> tmp_entities_;
  // Map for reference of Compuon
  std::unordered_map<const void *, std::any> entity_map_;

  // `thread_local` variable to save current session
  static thread_local std::shared_ptr<Session> session_ptr_;
};

} // namespace fhenomenon

#include "Session/Session.tpp"
