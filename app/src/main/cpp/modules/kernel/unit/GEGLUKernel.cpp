//
// Created by 구현우 on 2024/06/20.
//

#include "GEGLUKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "GEGLU_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }                          \

GEGLUKernel::GEGLUKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context,
            deviceId,
            assetManager,
            "kernel/geglu.cl"
    );

    gelu_multiply = clCreateKernel(program, "gelu_multiply", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

GEGLUKernel::~GEGLUKernel() {
    clReleaseKernel(gelu_multiply);
}