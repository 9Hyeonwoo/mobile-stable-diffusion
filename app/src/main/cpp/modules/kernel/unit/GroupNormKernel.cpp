//
// Created by 구현우 on 2024/06/20.
//

#include "GroupNormKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "GROUP_NORM_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }                          \

GroupNormKernel::GroupNormKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context,
            deviceId,
            assetManager,
            "kernel/group_norm.cl"
    );

    local_reduction_mean = clCreateKernel(program, "local_reduction_mean", &err);
    CHECK_ERROR_THROW(err);

    local_reduction_variance = clCreateKernel(program, "local_reduction_variance", &err);
    CHECK_ERROR_THROW(err);

    group_norm = clCreateKernel(program, "group_norm", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

GroupNormKernel::~GroupNormKernel() {
    clReleaseKernel(local_reduction_mean);
    clReleaseKernel(local_reduction_variance);
    clReleaseKernel(group_norm);
}