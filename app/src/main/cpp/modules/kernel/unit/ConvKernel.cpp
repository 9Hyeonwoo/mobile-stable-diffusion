//
// Created by 구현우 on 2024/06/20.
//

#include "ConvKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "CONV_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }                          \


ConvKernel::ConvKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context,
            deviceId,
            assetManager,
            "kernel/conv2d.cl"
    );

    conv2d = clCreateKernel(program, "conv2d", &err);
    CHECK_ERROR_THROW(err);

    im2col = clCreateKernel(program, "im2col", &err);
    CHECK_ERROR_THROW(err);

    conv2d_matmul = clCreateKernel(program, "conv2d_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win = clCreateKernel(program, "im2win", &err);
    CHECK_ERROR_THROW(err);

    im2win_matmul = clCreateKernel(program, "im2win_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_batch_matmul = clCreateKernel(program, "im2win_batch_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_reg_n_matmul = clCreateKernel(program, "im2win_reg_n_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_v2_matmul = clCreateKernel(program, "im2win_v2_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_matmul = clCreateKernel(program, "im2win_channel_reg_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_v4_matmul = clCreateKernel(program, "im2win_channel_reg_v4_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_transpose_v5_matmul = clCreateKernel(program, "im2win_channel_reg_transpose_v5_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_transpose = clCreateKernel(program, "im2win_transpose", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_transpose_vector_v6_matmul = clCreateKernel(program, "im2win_channel_reg_transpose_vector_v6_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_transpose_weight_vector_v7_matmul = clCreateKernel(program, "im2win_channel_reg_transpose_weight_vector_v7_matmul", &err);
    CHECK_ERROR_THROW(err);

    im2win_transpose_reorder = clCreateKernel(program, "im2win_transpose_reorder", &err);
    CHECK_ERROR_THROW(err);

    im2win_channel_reg_transpose_reorder_vector_v8_matmul = clCreateKernel(program, "im2win_channel_reg_transpose_reorder_vector_v8_matmul", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

ConvKernel::~ConvKernel() {
    clReleaseKernel(conv2d);
    clReleaseKernel(im2col);
    clReleaseKernel(conv2d_matmul);
    clReleaseKernel(im2win);
    clReleaseKernel(im2win_matmul);
    clReleaseKernel(im2win_batch_matmul);
    clReleaseKernel(im2win_reg_n_matmul);
    clReleaseKernel(im2win_v2_matmul);
    clReleaseKernel(im2win_channel_reg_matmul);
    clReleaseKernel(im2win_channel_reg_v4_matmul);
    clReleaseKernel(im2win_channel_reg_transpose_v5_matmul);
    clReleaseKernel(im2win_transpose);
    clReleaseKernel(im2win_channel_reg_transpose_vector_v6_matmul);
    clReleaseKernel(im2win_channel_reg_transpose_weight_vector_v7_matmul);
    clReleaseKernel(im2win_transpose_reorder);
    clReleaseKernel(im2win_channel_reg_transpose_reorder_vector_v8_matmul);
}