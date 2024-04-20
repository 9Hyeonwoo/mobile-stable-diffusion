//
// Created by 구현우 on 2024/04/20.
//

#include "LayerNormKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "LAYER_NORM_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

LayerNormKernel::LayerNormKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/layer_norm.cl");
    mean = clCreateKernel(program, "local_reduction_mean", &err);
    CHECK_ERROR_THROW(err);

    variance = clCreateKernel(program, "local_reduction_variance", &err);
    CHECK_ERROR_THROW(err);

    normalization = clCreateKernel(program, "layer_norm", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

LayerNormKernel::~LayerNormKernel() {
    clReleaseKernel(mean);
    clReleaseKernel(variance);
    clReleaseKernel(normalization);
}