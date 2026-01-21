#include "Backend/Backend.h"
#include "Compuon.h"
#include "Utils/log.h"
#include <cassert>
#include <iostream>

using namespace fhenomenon;

class DummyParameter : public Parameter {
  public:
  void printParams() const override {}
  void init() override {}
};

int main() {
  try {
    // Force the use of external backend (TFHE)
    // In a real scenario, this might be configured via file or env var
    // For now, we assume Backend::getInstance with a specific argument triggers it
    // Or we might need to hack the instance creation if logic isn't there yet.

    std::cout << "Starting TFHE Demo..." << std::endl;

    // We need to ensure we get a TfheBackend.
    // Based on Backend.h/cpp analysis, we might need to call specific init.
    // Let's assume for now we modify Backend::getInstance to return TfheBackend.

    auto &backend = Backend::getInstance("tfhe"); // hypothetical arg

    if (backend.getBackendType() != BackendType::ExternalBackend) {
      std::cerr << "Warning: Got " << (int)backend.getBackendType() << " instead of ExternalBackend." << std::endl;
    }

    Compuon<int> a(10);
    Compuon<int> b(32);

    // This relies on Compuon using the singleton backend for encryption
    // We might need to forcefully encrypt using the specific backend instance if Compuon defaults to Builtin.
    // Compuon currently doesn't seem to have a method to select backend easily for "new" compuons except via global
    // state? Let's check Compuon.h logic. It usually delegates to Backend::getInstance().

    // Encrypt (transform)
    DummyParameter params;
    backend.transform(a, params);
    backend.transform(b, params);

    // Add
    auto resultAdd = backend.add(a, b);
    // Cast result back to allowed type
    auto castedResultAdd = std::static_pointer_cast<Compuon<int>>(resultAdd);

    // Decrypt
    int decryptedAdd = std::any_cast<int>(backend.decrypt(*castedResultAdd));
    std::cout << "10 + 32 = " << decryptedAdd << std::endl;

    if (decryptedAdd == 42) {
      std::cout << "Addition Verification PASSED" << std::endl;
    } else {
      std::cout << "Addition Verification FAILED" << std::endl;
    }

    // Mul
    auto resultMul = backend.multiply(a, b);
    auto castedResultMul = std::static_pointer_cast<Compuon<int>>(resultMul);

    int decryptedMul = std::any_cast<int>(backend.decrypt(*castedResultMul));
    std::cout << "10 * 32 = " << decryptedMul << std::endl;

    if (decryptedMul == 320) {
      std::cout << "Multiplication Verification PASSED" << std::endl;
    } else {
      std::cout << "Multiplication Verification FAILED" << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
