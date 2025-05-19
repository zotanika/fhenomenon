#pragma once

#include "Parameter/Parameter.h"

namespace fhenomenon {

class Backend;

class Profile {
  public:
  std::shared_ptr<Parameter> getParam() { return param_; }

  void setParam(std::shared_ptr<Parameter> newParam) { param_.swap(newParam); }

  static std::shared_ptr<Profile> createProfile(std::shared_ptr<Parameter> newParam){
    profile_ = std::make_shared<Profile>();
    profile_->setParam(newParam);
    return profile_;
  }

  static std::shared_ptr<Profile> getProfile() {return profile_;}

  private:
  std::shared_ptr<Parameter> param_;
  static std::shared_ptr<Profile> profile_;
};

} // namespace fhenomenon
