//
// Created by 구현우 on 2024/04/20.
//

#include "LinearKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "LAYER_NORM_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

LinearKernel::LinearKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/linear.cl");

    naive_linear = clCreateKernel(program, "linear", &err);
    CHECK_ERROR_THROW(err);

    register_linear = clCreateKernel(program, "reg_linear", &err);
    CHECK_ERROR_THROW(err);
    clReleaseProgram(program);
}

LinearKernel::~LinearKernel() {
    clReleaseKernel(naive_linear);
    clReleaseKernel(register_linear);
}
