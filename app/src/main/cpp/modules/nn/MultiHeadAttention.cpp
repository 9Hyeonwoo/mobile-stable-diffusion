//
// Created by 구현우 on 2023/12/02.
//

#include "MultiHeadAttention.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "MULTI_HEAD_ATTENTION"
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

MultiHeadAttention::MultiHeadAttention(
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t embed_dim, size_t numHeads,
        const char *in_proj_weight_name,
        const char *in_proj_bias_name,
        const char *out_proj_weight_name,
        const char *out_proj_bias_name
) : context(context),
    cmdQueue(cmdQueue),
    numHeads(numHeads) {
    cl_int err;

    attnInProj0 = new Linear(context, cmdQueue, deviceId, assetManager,
                             embed_dim, embed_dim * 3,
                             in_proj_weight_name, in_proj_bias_name);

    attnOutProj0 = new Linear(context, cmdQueue, deviceId, assetManager,
                                embed_dim, embed_dim,
                              out_proj_weight_name, out_proj_bias_name);

    auto attention_mask = util::load_npy_file("encoder/attn_mask_fp32.npy");

    bufferAttentionMask = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         attention_mask.num_bytes(),
                                         nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferAttentionMask, CL_FALSE, 0,
                               attention_mask.num_bytes(),
                               attention_mask.data<float>(), 0, nullptr, nullptr);
    CHECK_ERROR_THROW(err);

    cl_program program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                                    "kernel/multi_head_attention.cl");

    kernel_add_matmul_attention = clCreateKernel(program, "add_matmul_attention", &err);
    CHECK_ERROR_THROW(err);

    kernel_softmax = clCreateKernel(program, "local_softmax", &err);
    CHECK_ERROR_THROW(err);

    kernel_matmul_attention = clCreateKernel(program, "batch_matmul_attention", &err);
    CHECK_ERROR_THROW(err);


    clReleaseProgram(program);

    program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                         "kernel/util.cl");

    kernel_permute3D_1_0_2 = clCreateKernel(program, "permute3D__1_0_2", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

MultiHeadAttention::~MultiHeadAttention() {
    delete attnInProj0;
    delete attnOutProj0;
    clReleaseMemObject(bufferAttentionMask);
    clReleaseKernel(kernel_add_matmul_attention);
    clReleaseKernel(kernel_softmax);
    clReleaseKernel(kernel_matmul_attention);
    clReleaseKernel(kernel_permute3D_1_0_2);
}

cl_int MultiHeadAttention::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                                   const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    size_t inputBytes;
    cl_event event1, event2, event3, event4, event5, event6, event7;
    cl_mem bufferEmbedding, bufferTemp, bufferAttnInProj0, bufferAttnInProj0_QKV, bufferAttentionQK;

    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    bufferEmbedding = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     inputBytes,
                                     nullptr, &err);
    CHECK_ERROR(err);

    bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                inputBytes,
                                nullptr, &err);
    CHECK_ERROR(err);

    bufferAttnInProj0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       inputBytes * 3,
                                       nullptr, &err);
    CHECK_ERROR(err);

    bufferAttnInProj0_QKV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           inputBytes * 3,
                                           nullptr, &err);
    CHECK_ERROR(err);

    bufferAttentionQK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * numHeads * CONTEXT_LENGTH * CONTEXT_LENGTH,
                                       nullptr, &err);
    CHECK_ERROR(err);

    /* self.model.transformer.resblocks[0].attn.in_proj Linear */
    err = attnInProj0->forward(input, bufferAttnInProj0, num_events_in_list, event_wait_list,
                               &event1);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(cmdQueue, bufferAttnInProj0, "encoder/test/resblock_0_attn_in_proj_test_fp32.npy");

    /* permute in_proj result to Q,K,V */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    CHECK_ERROR(err);

    size_t globalSizePermute_QKV[3] = {CONTEXT_LENGTH, 3, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 globalSizePermute_QKV, nullptr, 1,
                                 &event1,
                                 &event2);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(cmdQueue, bufferAttnInProj0_QKV, "encoder/test/resblock_0_attn_in_proj_qkv_test_fp32.npy");

    /* permute Q,K,V from (CONTEXT_LENGTH(77), BATCH(1)*NUM_HEADS(16), 1024/NUM_HEADS)
     * to (BATCH(1)*NUM_HEADS(16), CONTEXT_LENGTH(77), 1024/NUM_HEADS) */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0);
    CHECK_ERROR(err);

    size_t head_dim = EMBEDDING_SIZE / numHeads;
    size_t globalSizePermute_QKV_head[3] = {CONTEXT_LENGTH, numHeads, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 globalSizePermute_QKV_head, nullptr, 1,
                                 &event2,
                                 &event3);

    size_t globalOffsetK[3] = {CONTEXT_LENGTH, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, globalOffsetK,
                                  globalSizePermute_QKV_head, nullptr, 1,
                                  &event2,
                                  &event3);

    size_t globalOffsetV[3] = {CONTEXT_LENGTH * 2, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, globalOffsetV,
                                  globalSizePermute_QKV_head, nullptr, 1,
                                  &event2,
                                  &event3);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(cmdQueue, bufferAttnInProj0, "encoder/test/resblock_0_attn_in_proj_head_test_fp32.npy");

    /* scale dot attention - matmul QxK */

    err = clSetKernelArg(kernel_add_matmul_attention, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_add_matmul_attention, 1, sizeof(cl_mem), &bufferAttentionMask);
    err |= clSetKernelArg(kernel_add_matmul_attention, 2, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_add_matmul_attention, 3, sizeof(size_t), &head_dim);
    CHECK_ERROR(err);

    size_t globalSizeQK[3] = {numHeads, CONTEXT_LENGTH, CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_add_matmul_attention, 3, nullptr, globalSizeQK,
                                 nullptr, 3,
                                 &event3,
                                 &event4);

    // util::testBuffer(cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_qk_test_fp32.npy");

    /* scale dot attention - softmax */

    err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_softmax, 1, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_softmax, 2, sizeof(float) * CONTEXT_LENGTH, nullptr);
    CHECK_ERROR(err);

    size_t globalSizeSoftmax[1] = {numHeads * CONTEXT_LENGTH * CONTEXT_LENGTH};
    size_t localSizeSoftmax[1] = {CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_softmax, 1, nullptr, globalSizeSoftmax,
                                 localSizeSoftmax, 1,
                                 &event4,
                                 &event5);
    CHECK_ERROR(err);

    // util::testBuffer(cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_softmax_test_fp32.npy");

    /* scale dot attention - attention */

    size_t context_length = CONTEXT_LENGTH;
    err = clSetKernelArg(kernel_matmul_attention, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_matmul_attention, 1, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_matmul_attention, 2, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_matmul_attention, 3, sizeof(size_t), &context_length);
    CHECK_ERROR(err);

    size_t globalSizeMatmul[3] = {numHeads, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_matmul_attention, 3, nullptr, globalSizeMatmul,
                                 nullptr, 1,
                                 &event5,
                                 &event6);
    CHECK_ERROR(err);

    // max diff: 0.00000409036874771118
    // util::testBuffer(cmdQueue, bufferTemp, "encoder/test/resblock_0_attn_attention_test_fp32.npy");

    /* permute for input of out_proj */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSizePermuteOutProj[3] = {numHeads, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 globalSizePermuteOutProj, nullptr, 1,
                                 &event6,
                                 &event7);
    CHECK_ERROR(err);

    err = attnOutProj0->forward(bufferEmbedding, output, 1, &event7, event);
    CHECK_ERROR(err)

    // max diff: 0.00000362098217010498
    // util::testBuffer(cmdQueue, output, "encoder/test/resblock_0_attn_test_fp32.npy");

    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseEvent(event6);
    clReleaseEvent(event7);
    clReleaseMemObject(bufferEmbedding);
    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferAttnInProj0);
    clReleaseMemObject(bufferAttnInProj0_QKV);
    clReleaseMemObject(bufferAttentionQK);

    return CL_SUCCESS;
}
