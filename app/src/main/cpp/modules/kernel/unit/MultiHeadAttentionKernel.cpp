//
// Created by 구현우 on 2024/04/20.
//

#include "MultiHeadAttentionKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "MULTI_HEAD_ATTENTION_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

MultiHeadAttentionKernel::MultiHeadAttentionKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
        ) {
    cl_int err;

    cl_program program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                                    "kernel/multi_head_attention.cl");
    add_matmul_attention = clCreateKernel(program, "add_matmul_attention", &err);
    CHECK_ERROR_THROW(err);

    softmax = clCreateKernel(program, "local_softmax", &err);
    CHECK_ERROR_THROW(err);

    matmul_attention = clCreateKernel(program, "batch_matmul_attention", &err);
    CHECK_ERROR_THROW(err);

    batch_matmul_mask = clCreateKernel(program, "batch_matmul_mask", &err);
    CHECK_ERROR_THROW(err);

    batch_matmul = clCreateKernel(program, "batch_matmul", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

MultiHeadAttentionKernel::~MultiHeadAttentionKernel() {
    clReleaseKernel(add_matmul_attention);
    clReleaseKernel(softmax);
    clReleaseKernel(matmul_attention);
    clReleaseKernel(batch_matmul_mask);
    clReleaseKernel(batch_matmul);
}