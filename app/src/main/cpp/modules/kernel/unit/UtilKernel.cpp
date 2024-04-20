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

    gelu = clCreateKernel(program, "gelu", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

UtilKernel::~UtilKernel() {
    clReleaseKernel(elemwise_add);
    clReleaseKernel(permute3D_1_0_2);
    clReleaseKernel(gelu);
}