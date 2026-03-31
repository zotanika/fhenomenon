#include <gtest/gtest.h>

#include "FHN/fhn_backend_api.h"

#include <cstring>

TEST(FhnBackendApi, BackendInfoLayout) {
    FhnBackendInfo info;
    info.name = "test-backend";
    info.version = "1.0.0";
    info.device_type = FHN_DEVICE_GPU;
    info.device_memory = 8ULL * 1024 * 1024 * 1024;  /* 8 GiB */

    EXPECT_STREQ(info.name, "test-backend");
    EXPECT_STREQ(info.version, "1.0.0");
    EXPECT_EQ(info.device_type, FHN_DEVICE_GPU);
    EXPECT_EQ(info.device_memory, 8ULL * 1024 * 1024 * 1024);
}

static int dummy_kernel(FhnBackendCtx * /*ctx*/,
                        FhnBuffer * /*result*/,
                        const FhnBuffer *const * /*operands*/,
                        const int64_t * /*params*/,
                        const double * /*fparams*/) {
    return 0;
}

TEST(FhnBackendApi, KernelTableLayout) {
    FhnKernelEntry entry;
    entry.opcode = FHN_ADD_CC;
    entry.fn = dummy_kernel;
    entry.name = "add_cc";

    FhnKernelTable table;
    table.num_kernels = 1;
    table.kernels = &entry;

    EXPECT_EQ(table.num_kernels, 1u);
    EXPECT_EQ(table.kernels[0].opcode, FHN_ADD_CC);
    EXPECT_EQ(table.kernels[0].fn, dummy_kernel);
    EXPECT_STREQ(table.kernels[0].name, "add_cc");

    /* Verify the kernel function can be called and returns 0 */
    int rc = table.kernels[0].fn(nullptr, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(rc, 0);
}

TEST(FhnBackendApi, VTableNullOptional) {
    FhnBackendVTable vtable;
    std::memset(&vtable, 0, sizeof(vtable));

    /* Required fields are also NULL after zero-init, but the point is
       that optional fields are safely nullable. */
    EXPECT_EQ(vtable.get_info, nullptr);
    EXPECT_EQ(vtable.create, nullptr);
    EXPECT_EQ(vtable.destroy, nullptr);
    EXPECT_EQ(vtable.get_kernels, nullptr);

    /* Optional fields */
    EXPECT_EQ(vtable.submit, nullptr);
    EXPECT_EQ(vtable.poll, nullptr);
    EXPECT_EQ(vtable.wait, nullptr);
    EXPECT_EQ(vtable.get_outputs, nullptr);
    EXPECT_EQ(vtable.exec_free, nullptr);
}

TEST(FhnBackendApi, DeviceTypeEnum) {
    EXPECT_EQ(FHN_DEVICE_CPU, 0);
    EXPECT_EQ(FHN_DEVICE_GPU, 1);
    EXPECT_EQ(FHN_DEVICE_FPGA, 2);
    EXPECT_EQ(FHN_DEVICE_ASIC, 3);
}
