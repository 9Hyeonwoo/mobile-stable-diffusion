//
// Created by 구현우 on 2024/06/20.
//

#include "CrossAttentionKernel.h"
#include "../../util.h"
#include <android/log.h>

#define LOG_TAG "CROSS_ATTENTION_KERNEL"

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }                          \


CrossAttentionKernel::CrossAttentionKernel(
        cl_context context,
        cl_device_id deviceId,
        AAssetManager *assetManager
) {
    cl_int err;

    auto program = util::create_and_build_program_with_source(
            context,
            deviceId,
            assetManager,
            "kernel/cross_attention.cl"
    );

    einsum_bik_bjk_bij = clCreateKernel(program, "einsum_bik_bjk_bij", &err);
    CHECK_ERROR_THROW(err);

    einsum_bij_bjk_bik = clCreateKernel(program, "einsum_bij_bjk_bik", &err);
    CHECK_ERROR_THROW(err);

    optimized_einsum_bik_bjk_bij = clCreateKernel(program, "optimized_einsum_bik_bjk_bij", &err);
    CHECK_ERROR_THROW(err);

    optimized_einsum_bik_bkj_bij = clCreateKernel(program, "optimized_einsum_bik_bkj_bij", &err);
    CHECK_ERROR_THROW(err);

    optimized_einsum_bik_bkj_bij_general = clCreateKernel(program, "optimized_einsum_bik_bkj_bij_general", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

CrossAttentionKernel::~CrossAttentionKernel() {
    clReleaseKernel(einsum_bik_bjk_bij);
    clReleaseKernel(einsum_bij_bjk_bik);
    clReleaseKernel(optimized_einsum_bik_bjk_bij);
    clReleaseKernel(optimized_einsum_bik_bkj_bij);
    clReleaseKernel(optimized_einsum_bik_bkj_bij_general);
}