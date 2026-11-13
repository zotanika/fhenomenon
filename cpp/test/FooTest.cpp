#include "Foo.hpp"
#include <gtest/gtest.h>

TEST(FooTest, SetName) {
  Example::Foo foo;
  foo.setName("bar");
  EXPECT_EQ(foo.getName(), "bar");
}
