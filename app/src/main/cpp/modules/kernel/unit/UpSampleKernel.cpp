//
// Created by 구현우 on 2024/06/20.
//

#include "UpSampleKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "UP_SAMPLE_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }                          \

UpSampleKernel::UpSampleKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context,
            deviceId,
            assetManager,
            "kernel/up_sample.cl"
    );

    up_sample_nearest = clCreateKernel(program, "up_sample_nearest", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

UpSampleKernel::~UpSampleKernel() {
    clReleaseKernel(up_sample_nearest);
}