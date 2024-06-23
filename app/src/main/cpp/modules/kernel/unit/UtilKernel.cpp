//
// Created by 구현우 on 2024/04/20.
//

#include "UtilKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "UTIL_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

UtilKernel::UtilKernel(cl_context context, cl_device_id deviceId, AAssetManager *assetManager) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context, deviceId, assetManager, "kernel/util.cl"
    );

    elemwise_add = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR_THROW(err);

    permute3D_1_0_2 = clCreateKernel(program, "permute3D__1_0_2", &err);
    CHECK_ERROR_THROW(err);

    permute3D_0_2_1 = clCreateKernel(program, "permute3D__0_2_1", &err);
    CHECK_ERROR_THROW(err);

    permute3D = clCreateKernel(program, "permute3D", &err);
    CHECK_ERROR_THROW(err);

    gelu = clCreateKernel(program, "gelu", &err);
    CHECK_ERROR_THROW(err);

    softmax = clCreateKernel(program, "softmax", &err);
    CHECK_ERROR_THROW(err);

    silu = clCreateKernel(program, "silu", &err);
    CHECK_ERROR_THROW(err);

    batch_matmul = clCreateKernel(program, "batch_matmul", &err);
    CHECK_ERROR_THROW(err);

    batch_matmul_scale = clCreateKernel(program, "batch_matmul_scale", &err);
    CHECK_ERROR_THROW(err);

    chunkwise_add = clCreateKernel(program, "chunkwise_add", &err);
    CHECK_ERROR_THROW(err);

    permute3D_copy = clCreateKernel(program, "permute3D_copy", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

UtilKernel::~UtilKernel() {
    clReleaseKernel(elemwise_add);
    clReleaseKernel(permute3D_1_0_2);
    clReleaseKernel(permute3D_0_2_1);
    clReleaseKernel(permute3D);
    clReleaseKernel(gelu);
    clReleaseKernel(softmax);
    clReleaseKernel(silu);
    clReleaseKernel(batch_matmul);
    clReleaseKernel(batch_matmul_scale);
    clReleaseKernel(chunkwise_add);
    clReleaseKernel(permute3D_copy);
}