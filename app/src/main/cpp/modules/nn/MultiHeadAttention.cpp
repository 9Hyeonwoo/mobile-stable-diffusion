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
        size_t embed_dim, size_t numHeads,
        const std::string &in_proj_weight_name,
        const std::string &in_proj_bias_name,
        const std::string &out_proj_weight_name,
        const std::string &out_proj_bias_name,
        cl_mem attentionMask,
        LinearKernel &linearKernel,
        MultiHeadAttentionKernel &kernel,
        UtilKernel &utilKernel
) : context(context),
    cmdQueue(cmdQueue),
    numHeads(numHeads),
    kernel(kernel), utilKernel(utilKernel) {

    attnInProj0 = new Linear(context, cmdQueue,
                             embed_dim, embed_dim * 3,
                             in_proj_weight_name, in_proj_bias_name,
                             linearKernel);

    attnOutProj0 = new Linear(context, cmdQueue,
                                embed_dim, embed_dim,
                              out_proj_weight_name, out_proj_bias_name,
                              linearKernel);

    bufferAttentionMask = attentionMask;
}

MultiHeadAttention::~MultiHeadAttention() {
    delete attnInProj0;
    delete attnOutProj0;
}

void MultiHeadAttention::init() {
    attnInProj0->init();
    attnOutProj0->init();
}

cl_int MultiHeadAttention::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                                   const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    size_t inputBytes;
    cl_event event1, event2, event3, event4, event5, event6, event7;
    cl_mem bufferEmbedding, bufferTemp, bufferAttnInProj0, bufferAttnInProj0_QKV, bufferAttentionQK;
    cl_mem bufferQ, bufferK, bufferV;

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
    err = clSetKernelArg(utilKernel.permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(utilKernel.permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    CHECK_ERROR(err);

    size_t globalSizePermute_QKV[3] = {CONTEXT_LENGTH, 3, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel.permute3D_1_0_2, 3, nullptr,
                                 globalSizePermute_QKV, nullptr, 1,
                                 &event1,
                                 &event2);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(cmdQueue, bufferAttnInProj0_QKV, "encoder/test/resblock_0_attn_in_proj_qkv_test_fp32.npy");

    /* permute Q,K,V from (CONTEXT_LENGTH(77), BATCH(1)*NUM_HEADS(16), 1024/NUM_HEADS)
     * to (BATCH(1)*NUM_HEADS(16), CONTEXT_LENGTH(77), 1024/NUM_HEADS) */
    err = clSetKernelArg(utilKernel.permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    err |= clSetKernelArg(utilKernel.permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0);
    CHECK_ERROR(err);

    size_t head_dim = EMBEDDING_SIZE / numHeads;
    size_t globalSizePermute_QKV_head[3] = {CONTEXT_LENGTH, numHeads, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel.permute3D_1_0_2, 3, nullptr,
                                 globalSizePermute_QKV_head, nullptr, 1,
                                 &event2,
                                 &event3);

    size_t globalOffsetK[3] = {CONTEXT_LENGTH, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, utilKernel.permute3D_1_0_2, 3, globalOffsetK,
                                  globalSizePermute_QKV_head, nullptr, 1,
                                  &event2,
                                  &event3);

    size_t globalOffsetV[3] = {CONTEXT_LENGTH * 2, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, utilKernel.permute3D_1_0_2, 3, globalOffsetV,
                                  globalSizePermute_QKV_head, nullptr, 1,
                                  &event2,
                                  &event3);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(cmdQueue, bufferAttnInProj0, "encoder/test/resblock_0_attn_in_proj_head_test_fp32.npy");

    /* scale dot attention - matmul QxK */
    cl_buffer_region regionQ = {0, sizeof(float) * numHeads * CONTEXT_LENGTH * head_dim};
    bufferQ = clCreateSubBuffer(bufferAttnInProj0, CL_MEM_READ_ONLY,
                                CL_BUFFER_CREATE_TYPE_REGION, &regionQ, &err);
    CHECK_ERROR(err);

    cl_buffer_region regionK = {sizeof(float) * numHeads * CONTEXT_LENGTH * head_dim, sizeof(float) * numHeads * CONTEXT_LENGTH * head_dim};
    bufferK = clCreateSubBuffer(bufferAttnInProj0, CL_MEM_READ_ONLY,
                                CL_BUFFER_CREATE_TYPE_REGION, &regionK, &err);
    CHECK_ERROR(err);

    cl_buffer_region regionV = {sizeof(float) * numHeads * CONTEXT_LENGTH * head_dim * 2, sizeof(float) * numHeads * CONTEXT_LENGTH * head_dim};
    bufferV = clCreateSubBuffer(bufferAttnInProj0, CL_MEM_READ_ONLY,
                                CL_BUFFER_CREATE_TYPE_REGION, &regionV, &err);
    CHECK_ERROR(err);

    /* naive QxK
    err = clSetKernelArg(kernel.add_matmul_attention, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel.add_matmul_attention, 1, sizeof(cl_mem), &bufferAttentionMask);
    err |= clSetKernelArg(kernel.add_matmul_attention, 2, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.add_matmul_attention, 3, sizeof(size_t), &head_dim);
    CHECK_ERROR(err);

    size_t globalSizeQK[3] = {numHeads, CONTEXT_LENGTH, CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.add_matmul_attention, 3, nullptr, globalSizeQK,
                                 nullptr, 3,
                                 &event3,
                                 &event4);
     naive QxK */

    /* optimized QxK */
    size_t MN = CONTEXT_LENGTH;
    size_t reg_size = 7, tile_size = 77;
    err = clSetKernelArg(kernel.batch_matmul_mask, 0, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 1, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 2, sizeof(cl_mem), &bufferAttentionMask);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 3, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 4, sizeof(size_t), &MN);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 5, sizeof(size_t), &MN);
    err |= clSetKernelArg(kernel.batch_matmul_mask, 6, sizeof(size_t), &head_dim);
    CHECK_ERROR(err);

    size_t globalSizeQK[3] = {numHeads,MN/reg_size, MN/reg_size};
    size_t localSizeQK[3] = {1, tile_size/reg_size, tile_size/reg_size};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.batch_matmul_mask, 3, nullptr, globalSizeQK,
                                 localSizeQK, 3,
                                 &event3,
                                 &event4);
    CHECK_ERROR(err);
    /* optimized QxK */


    // util::testBuffer(cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_qk_test_fp32.npy");

    /* scale dot attention - softmax */

    err = clSetKernelArg(kernel.softmax, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.softmax, 1, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.softmax, 2, sizeof(float) * CONTEXT_LENGTH, nullptr);
    CHECK_ERROR(err);

    size_t globalSizeSoftmax[1] = {numHeads * CONTEXT_LENGTH * CONTEXT_LENGTH};
    size_t localSizeSoftmax[1] = {CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.softmax, 1, nullptr, globalSizeSoftmax,
                                 localSizeSoftmax, 1,
                                 &event4,
                                 &event5);
    CHECK_ERROR(err);

    // util::testBuffer(cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_softmax_test_fp32.npy");

    /* scale dot attention - attention */

    /* naive - QK x V
    size_t context_length = CONTEXT_LENGTH;
    err = clSetKernelArg(kernel.matmul_attention, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.matmul_attention, 1, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel.matmul_attention, 2, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel.matmul_attention, 3, sizeof(size_t), &context_length);
    CHECK_ERROR(err);

    size_t globalSizeMatmul[3] = {numHeads, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.matmul_attention, 3, nullptr, globalSizeMatmul,
                                 nullptr, 1,
                                 &event5,
                                 &event6);
    CHECK_ERROR(err);
     naive - QK x V*/

    /* optimized - QK x V */
    size_t tile_size1 = 64, reg_size1 = 8;
    err = clSetKernelArg(kernel.batch_matmul, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel.batch_matmul, 1, sizeof(cl_mem), &bufferV);
    err |= clSetKernelArg(kernel.batch_matmul, 2, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel.batch_matmul, 3, sizeof(size_t), &MN);
    err |= clSetKernelArg(kernel.batch_matmul, 4, sizeof(size_t), &head_dim);
    err |= clSetKernelArg(kernel.batch_matmul, 5, sizeof(size_t), &MN);
    CHECK_ERROR(err);

    size_t globalSizeMatmul[3] = {numHeads,MN/reg_size, head_dim/reg_size1};
    size_t localSizeMatmul[3] = {1, tile_size/reg_size, tile_size1/reg_size1};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel.batch_matmul, 3, nullptr, globalSizeMatmul,
                                 localSizeMatmul, 1,
                                 &event5,
                                 &event6);
    CHECK_ERROR(err);
    /* optimized - QK x V */

    // max diff: 0.00000409036874771118
    // util::testBuffer(cmdQueue, bufferTemp, "encoder/test/resblock_0_attn_attention_test_fp32.npy");

    /* permute for input of out_proj */
    err = clSetKernelArg(utilKernel.permute3D_1_0_2, 0, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(utilKernel.permute3D_1_0_2, 1, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSizePermuteOutProj[3] = {numHeads, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel.permute3D_1_0_2, 3, nullptr,
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
    clReleaseMemObject(bufferQ);
    clReleaseMemObject(bufferK);
    clReleaseMemObject(bufferV);

    return CL_SUCCESS;
}
