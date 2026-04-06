#include "Backend/External.h"

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>

namespace fhenomenon {

ExternalBackend::ExternalBackend(const std::string &libraryPath,
                                 const char *config_json,
                                 const std::string &symbolPrefix) {
  // 1. dlopen the shared library
  dl_handle_ = dlopen(libraryPath.c_str(), RTLD_LAZY);
  if (!dl_handle_) {
    throw std::runtime_error(std::string("ExternalBackend: dlopen failed: ") +
                             dlerror());
  }

  // Build symbol names with optional prefix
  auto sym = [&](const char *base) { return symbolPrefix + base; };

  // 2. Resolve the 4 required symbols
  vtable_.get_info = reinterpret_cast<FhnGetInfoFn>(
      dlsym(dl_handle_, sym("fhn_get_info").c_str()));
  vtable_.create = reinterpret_cast<FhnCreateFn>(
      dlsym(dl_handle_, sym("fhn_create").c_str()));
  vtable_.destroy = reinterpret_cast<FhnDestroyFn>(
      dlsym(dl_handle_, sym("fhn_destroy").c_str()));
  vtable_.get_kernels = reinterpret_cast<FhnGetKernelsFn>(
      dlsym(dl_handle_, sym("fhn_get_kernels").c_str()));

  if (!vtable_.get_info || !vtable_.create || !vtable_.destroy ||
      !vtable_.get_kernels) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error(
        "ExternalBackend: library missing required fhn_* symbols");
  }

  // 3. Resolve optional advanced symbols (NULL if absent)
  vtable_.submit = reinterpret_cast<FhnSubmitFn>(
      dlsym(dl_handle_, sym("fhn_submit").c_str()));
  vtable_.poll = reinterpret_cast<FhnPollFn>(
      dlsym(dl_handle_, sym("fhn_poll").c_str()));
  vtable_.wait = reinterpret_cast<FhnWaitFn>(
      dlsym(dl_handle_, sym("fhn_wait").c_str()));
  vtable_.get_outputs = reinterpret_cast<FhnGetOutputsFn>(
      dlsym(dl_handle_, sym("fhn_get_outputs").c_str()));
  vtable_.exec_free = reinterpret_cast<FhnExecFreeFn>(
      dlsym(dl_handle_, sym("fhn_exec_free").c_str()));

  // 4. Get backend info
  info_ = vtable_.get_info();

  // 5. Create backend context
  fhn_ctx_ = vtable_.create(config_json);
  if (!fhn_ctx_) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: fhn_create returned null");
  }

  // 6. Get kernel table and create executor
  fhn_table_ = vtable_.get_kernels(fhn_ctx_);
  if (!fhn_table_) {
    vtable_.destroy(fhn_ctx_);
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: fhn_get_kernels returned null");
  }

  executor_ = std::make_unique<FhnDefaultExecutor>(fhn_table_);

  std::cout << "ExternalBackend loaded: " << info_->name << " v"
            << info_->version
            << " (device_type=" << static_cast<int>(info_->device_type) << ")"
            << std::endl;
}

ExternalBackend::~ExternalBackend() {
  executor_.reset();
  if (fhn_ctx_ && vtable_.destroy) {
    vtable_.destroy(fhn_ctx_);
    fhn_ctx_ = nullptr;
  }
  if (dl_handle_) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
  }
}

void ExternalBackend::transform(CompuonBase & /*entity*/,
                                const Parameter & /*params*/) const {
  // External backends handle encryption through FHN_ENCRYPT kernel
}

std::shared_ptr<CompuonBase>
ExternalBackend::add(const CompuonBase & /*a*/,
                     const CompuonBase & /*b*/) const {
  // Direct Backend::add() is legacy interface.
  // External backends use FHN executor pipeline.
  return nullptr;
}

std::shared_ptr<CompuonBase>
ExternalBackend::multiply(const CompuonBase & /*a*/,
                          const CompuonBase & /*b*/) const {
  return nullptr;
}

std::any ExternalBackend::decrypt(const CompuonBase & /*entity*/) const {
  return 0;
}

} // namespace fhenomenon
