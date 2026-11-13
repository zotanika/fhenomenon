#include <string>

namespace Example {

class Foo final {
  public:
  void setName(const std::string &name);
  const std::string &getName() const;

  private:
  std::string name_;
};

} // namespace Example
