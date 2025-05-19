#include "Foo.hpp"

namespace Example {

void Foo::setName(const std::string &name) { name_ = name; }

const std::string &Foo::getName() const { return name_; }

} // namespace Example
