#include <FHN/Runtime.h>
#include <array>

// A separate contribution using only the installed public runtime header.
// It composes two commands sharing one caller-owned preparation artifact.
void registerRotationPair(fhenomenon::runtime::Runtime &runtime) {
  using namespace fhenomenon::runtime;
  runtime.addRecipe({"example.rotation-pair", "installed.two-rotation", 0,
                     [](const Request &r, std::string &reason) {
                       if (r.inputs.size() != 2 || !std::any_cast<std::array<std::int64_t, 2>>(&r.attributes)) {
                         reason = "requires source, preparation, and two public rotation amounts";
                         return false;
                       }
                       return true;
                     },
                     [](Runtime &rt, const Request &r) {
                       const auto amounts = std::any_cast<std::array<std::int64_t, 2>>(r.attributes);
                       auto first = rt.submit({"native.rotate", r.inputs, amounts[0]});
                       auto second = rt.submit({"native.rotate", r.inputs, amounts[1]});
                       return Values{first.at(0), second.at(0)};
                     }});
}
