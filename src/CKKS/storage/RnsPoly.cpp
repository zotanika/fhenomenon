#include "CKKS/RnsPoly.h"

#include <cstring>

namespace fhenomenon::ckks {

bool copy(RnsPolyRef dst, RnsPolyView src) {
  if (dst.layout() != src.layout()) {
    return false;
  }
  if (dst.data() == src.data()) {
    return true;
  }
  // memmove rather than memcpy: the ABI lets a kernel's result alias an
  // operand, and a copy is the one operation where that alias is total.
  std::memmove(dst.data(), src.data(), src.words() * sizeof(uint64_t));
  return true;
}

} // namespace fhenomenon::ckks
