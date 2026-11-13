#include "FHN/contrib/PublicLinearKernels.h"
#include "support/Client.h"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace fhenomenon::runtime;
using fhenomenon::contrib::PublicLinearRuntime;
using fhenomenon::contrib::registerPublicLinearKernels;
TEST(PublicLinearContribution, ReusesPublicRowWithDifferentEncryptedInputsAndOneLevel) {
  Runtime runtime;
  test_support::Client client(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  registerPublicLinearKernels(runtime, client.services(), FHN_NATIVE_TEST_LIBRARY);
  PublicLinearRuntime linear(runtime);
  auto first = client.encrypt({0.5, -1, 2, 0});
  auto second = client.encrypt({2, 1, -0.5, 0});
  auto row = linear.prepareRow(first, {0.25, -2, 0.5, 0});
  EXPECT_TRUE(row.dependencies().empty());
  auto a = linear.dot(first, row, 0.75);
  auto b = linear.dot(second, row, -0.25);
  for (auto value : client.decrypt(a))
    EXPECT_NEAR(value, 3.875, 1e-5);
  for (auto value : client.decrypt(b))
    EXPECT_NEAR(value, -2, 1e-5);
  NativeRuntime native(runtime, client.services());
  FhnCkksBufferInfo before{}, after{};
  ASSERT_EQ(client.services().buffer_info(native.context(), native.buffer(first), &before), 0);
  ASSERT_EQ(client.services().buffer_info(native.context(), native.buffer(a), &after), 0);
  EXPECT_EQ(after.level, before.level - 1);
  EXPECT_NEAR(client.decrypt(first)[1], -1, 1e-7);
}
TEST(PublicLinearContribution, ScopePreservesEveryOutputAndSharedInput) {
  Runtime runtime;
  test_support::Client client(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  registerPublicLinearKernels(runtime, client.services(), FHN_NATIVE_TEST_LIBRARY);
  PublicLinearRuntime linear(runtime);
  auto input = client.encrypt({1, 2, 3, 4});
  auto row = linear.prepareRow(input, {0.5, 0.25, 0, -0.25});
  auto outputs = runtime.runScope([&](Scope &scope) {
    auto x = scope.input(input), w = scope.input(row);
    auto a = linear.dot(scope, x, w, 1);
    auto b = linear.dot(scope, x, w, 2);
    return ScopeValues{b, x, a, b};
  });
  ASSERT_EQ(outputs.size(), 4u);
  EXPECT_EQ(outputs[0].identity(), outputs[3].identity());
  for (auto value : client.decrypt(outputs[0]))
    EXPECT_NEAR(value, 2, 1e-5);
  for (auto value : client.decrypt(outputs[2]))
    EXPECT_NEAR(value, 1, 1e-5);
  EXPECT_NEAR(client.decrypt(outputs[1])[3], 4, 1e-7);
}
TEST(PublicLinearContribution, RejectsWrongRowsBiasAndContext) {
  Runtime runtime;
  test_support::Client client(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  registerPublicLinearKernels(runtime, client.services(), FHN_NATIVE_TEST_LIBRARY);
  PublicLinearRuntime linear(runtime);
  auto input = client.encrypt({1, 2, 3, 4});
  EXPECT_THROW(linear.prepareRow(input, {1, 2}), std::exception);
  EXPECT_THROW(linear.prepareRow(input, {1, 2, 3, std::numeric_limits<double>::infinity()}), std::exception);
  auto row = linear.prepareRow(input, {1, 0, 0, 0});
  EXPECT_THROW(linear.dot(input, row, std::numeric_limits<double>::quiet_NaN()), std::exception);
  test_support::Client other(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  auto foreign = other.encrypt({1, 2, 3, 4});
  EXPECT_THROW(linear.dot(foreign, row), std::exception);
  for (auto value : client.decrypt(linear.dot(input, row)))
    EXPECT_NEAR(value, 1, 1e-5);
}
TEST(PublicLinearContribution, RejectsExtensionFromDifferentServicesBeforeRegistration) {
  Runtime runtime;
  test_support::Client client(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  auto services = client.services();
  services.buffer_info = [](FhnBackendCtx *, const FhnBuffer *, FhnCkksBufferInfo *) { return -1; };
  EXPECT_THROW(registerPublicLinearKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY), Error);
}
TEST(PublicLinearContribution, RejectsCopiedPreparationAndAccountsPublicationBuffer) {
  Runtime runtime;
  test_support::Client client(runtime, FHN_NATIVE_TEST_LIBRARY, 4);
  registerPublicLinearKernels(runtime, client.services(), FHN_NATIVE_TEST_LIBRARY);
  PublicLinearRuntime linear(runtime);
  auto input = client.encrypt({1, 2, 3, 4});
  auto row = linear.prepareRow(input, {1, 0, 0, 0});
  auto forged = runtime.makeValue(row.format(), row.kind(), row.bytes(), [&]() { return row.payload(); });
  EXPECT_THROW(linear.dot(input, forged), Error);

  Runtime limited({input.bytes() + row.bytes(), 0});
  NativeRuntime native(runtime, client.services()), limitedNative(limited, client.services());
  registerPublicLinearKernels(limited, client.services(), FHN_NATIVE_TEST_LIBRARY);
  PublicLinearRuntime limitedLinear(limited);
  auto limitedInput = limitedNative.importCopy(native.buffer(input));
  auto limitedRow = limitedLinear.prepareRow(limitedInput, {1, 0, 0, 0});
  const auto retained = limited.stats().live_bytes;
  try {
    (void)limitedLinear.dot(limitedInput, limitedRow);
    FAIL() << "publication must obey retained-resource limit";
  } catch (const Error &error) {
    EXPECT_EQ(error.code(), ErrorCode::BudgetExceeded);
  }
  EXPECT_EQ(limited.stats().live_bytes, retained);
  EXPECT_NEAR(client.decrypt(input)[0], 1, 1e-7);
}
} // namespace
