//
// Created by 구현우 on 2023/12/02.
//

#include "ResidualAttentionBlock.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "RESIDUAL_ATTENTION_BLOCK"
#define CONTEXT_LENGTH 77
#define EMBEDDING_SIZE 1024

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      return err;                     \
    }                    \

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

ResidualAttentionBlock::ResidualAttentionBlock(
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t numHeads
) : context(context),
    cmdQueue(cmdQueue),
    assetManager(assetManager) {
    cl_int err;
    ln_1 = new LayerNorm(context, cmdQueue, deviceId, assetManager,
                         "encoder/resblock_0_ln_1_weight_fp32.npy",
                         "encoder/resblock_0_ln_1_bias_fp32.npy");

    ln_2 = new LayerNorm(context, cmdQueue, deviceId, assetManager,
                         "encoder/resblock_0_ln_2_weight_fp32.npy",
                         "encoder/resblock_0_ln_2_bias_fp32.npy");

    attn = new MultiHeadAttention(context, cmdQueue, deviceId, assetManager, numHeads);

    mlp_c_fc = new Linear(context, cmdQueue, deviceId, assetManager,
                          "encoder/resblock_0_mlp_c_fc_weight_fp32.npy",
                          "encoder/resblock_0_mlp_c_fc_bias_fp32.npy");

    mlp_c_proj = new Linear(context, cmdQueue, deviceId, assetManager,
                          "encoder/resblock_0_mlp_c_proj_weight_fp32.npy",
                          "encoder/resblock_0_mlp_c_proj_bias_fp32.npy");

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/util.cl");

    kernel_elemwise_add = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR_THROW(err);

    kernel_gelu = clCreateKernel(program, "gelu", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

ResidualAttentionBlock::~ResidualAttentionBlock() {
    delete ln_1;
    delete ln_2;
    delete attn;
    delete mlp_c_fc;
    delete mlp_c_proj;
    clReleaseKernel(kernel_elemwise_add);
    clReleaseKernel(kernel_gelu);
}

cl_int ResidualAttentionBlock::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    size_t inputBytes, inputSize;
    cl_event event1, event2, event3, event4, event5, event6, event7;
    cl_mem bufferEmbedding, bufferTemp, bufferMLP;

    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    inputSize = inputBytes / sizeof(float);

    bufferEmbedding = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     inputBytes,
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                inputBytes,
                                nullptr, &err);
    CHECK_ERROR(err);

    bufferMLP = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               inputBytes * 4,
                               nullptr, &err);
    CHECK_ERROR(err);

    err = ln_1->forward(input, bufferEmbedding, num_events_in_list, event_wait_list, &event1);
    CHECK_ERROR(err);

    err = attn->forward(bufferEmbedding, bufferEmbedding, 1, &event1, &event2);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_elemwise_add, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_elemwise_add, 1, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(kernel_elemwise_add, 2, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSize[] = {inputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_elemwise_add, 1, nullptr, globalSize, nullptr, 1,
                                 &event2, &event3);
    CHECK_ERROR(err);

    // max diff: 0.00000362098217010498
    // util::testBuffer(assetManager, cmdQueue, bufferEmbedding, "encoder/test/resblock_0_add_attn_test_fp32.npy");

    err = ln_2->forward(bufferEmbedding, bufferTemp, 1, &event3, &event4);

    // max diff: 0.00003504753112792969
    // util::testBuffer(assetManager, cmdQueue, bufferTemp, "encoder/test/resblock_0_ln2_test_fp32.npy");

    err = mlp_c_fc->forward(bufferTemp, bufferMLP, 1, &event4, &event5);

    // max diff: 0.00002098083496093750
    // util::testBuffer(assetManager, cmdQueue, bufferMLP, "encoder/test/resblock_0_mlp_c_fc_test_fp32.npy");

    err = clSetKernelArg(kernel_gelu, 0, sizeof(cl_mem), &bufferMLP);
    err |= clSetKernelArg(kernel_gelu, 1, sizeof(cl_mem), &bufferMLP);
    CHECK_ERROR(err);

    size_t globalSizeGELU[] = {inputSize * 4};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_gelu, 1, nullptr, globalSizeGELU, nullptr, 1,
                                 &event5, &event6);
    CHECK_ERROR(err);

    // max diff: 0.00002098083496093750
    // util::testBuffer(assetManager, cmdQueue, bufferMLP, "encoder/test/resblock_0_mlp_gelu_test_fp32.npy");

    err = mlp_c_proj->forward(bufferMLP, bufferTemp, 1, &event6, &event7);
    CHECK_ERROR(err);

    // max diff: 0.00003051757812500000
    // util::testBuffer(assetManager, cmdQueue, bufferTemp, "encoder/test/resblock_0_mlp_c_proj_test_fp32.npy");

    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseEvent(event6);
    clReleaseEvent(event7);
    clReleaseMemObject(bufferEmbedding);
    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferMLP);

    return CL_SUCCESS;
}