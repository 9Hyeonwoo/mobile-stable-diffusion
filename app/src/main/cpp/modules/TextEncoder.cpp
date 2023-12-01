//
// Created by 구현우 on 2023/11/24.
//

#include "TextEncoder.h"
#include "util.h"

#include <chrono>

#define LOG_TAG "TEXT_ENCODER"
#define EMBEDDING_SIZE 1024
#define NUM_HEADS 16
#define CONTEXT_LENGTH 77

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

#define PRINT_TIME(index, ...) \
    do { \
        auto start##index = std::chrono::steady_clock::now(); \
        { __VA_ARGS__; } \
        auto end##index = std::chrono::steady_clock::now(); \
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "PRINT_TIME[%s]: %lld", #index, std::chrono::duration_cast<std::chrono::milliseconds>(end##index - start##index).count()); \
    } while(0)

TextEncoder::TextEncoder(AAssetManager *assetManager, cl_context context, cl_command_queue cmdQueue,
                         cl_device_id deviceId) : context(context), cmdQueue(cmdQueue),
                                                  deviceId(deviceId), assetManager(assetManager) {
    cl_int err;
    embedding = util::load_npy_file(assetManager, "encoder/embedding_fp32.npy");
    auto positional_embedding = util::load_npy_file(assetManager, "encoder/positional_embedding_fp32.npy");
    auto attention_mask = util::load_npy_file(assetManager, "encoder/attn_mask_fp32.npy");

    bufferPositionalEmbedding = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       positional_embedding.num_bytes(),
                                       nullptr, &err);
    CHECK_ERROR(err);

    bufferAttentionMask = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                               attention_mask.num_bytes(),
                                               nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferPositionalEmbedding, CL_TRUE, 0,
                         positional_embedding.num_bytes(),
                         positional_embedding.data<float>(), 0, nullptr, nullptr);

    err |= clEnqueueWriteBuffer(cmdQueue, bufferAttentionMask, CL_TRUE, 0,
                                attention_mask.num_bytes(),
                                attention_mask.data<float>(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    layerNorm0 = new LayerNorm(context, cmdQueue, deviceId, assetManager,
                               "encoder/layer_norm_0_weight_fp32.npy",
                               "encoder/layer_norm_0_bias_fp32.npy");

    attnInProj0 = new Linear(context, cmdQueue, deviceId, assetManager,
                             "encoder/resblock_0_attn_in_proj_weight_fp32.npy",
                             "encoder/resblock_0_attn_in_proj_bias_fp32.npy");

    attnOutProj0 = new Linear(context, cmdQueue, deviceId, assetManager,
                             "encoder/resblock_0_attn_out_proj_weight_fp32.npy",
                             "encoder/resblock_0_attn_out_proj_bias_fp32.npy");
}

TextEncoder::~TextEncoder() {
    delete layerNorm0;
    delete attnInProj0;
    delete attnOutProj0;
    clReleaseMemObject(bufferPositionalEmbedding);
    clReleaseMemObject(bufferAttentionMask);
}

/*
 * @input: `token` tokenized text
 * @return: token embedding with size=(length of 'token'(CONTEXT_LENGTH=77) * (EMBEDDING_SIZE=1024))
 */
std::vector<float> TextEncoder::token_embedding(const std::vector<long> &token) {
    std::vector<float> result;
    for (auto i: token) {
        auto data = embedding.data<float>() + (i * EMBEDDING_SIZE);
        result.insert(result.end(), data, data + EMBEDDING_SIZE);
    }
    return result;
}

std::vector<float> TextEncoder::encode(const std::vector<long> &token) {
    cl_int err;
    cl_event event1, event2, event3, event4, event5, event6, event7, event8, event9, event10, event11;
    cl_kernel kernel_elemwise_add, kernel_permute3D_1_0_2;
//    testEmbedding(token);
//    PRINT_TIME(0,
    std::vector<float> token_embedding_result = token_embedding(token);

    // elemwise_add
    cl_mem bufferEmbedding = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(float) * token_embedding_result.size(),
                                            nullptr, &err);
    CHECK_ERROR(err);
    cl_mem bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * token_embedding_result.size(),
                                       nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferEmbedding, CL_FALSE, 0,
                         sizeof(float) * token_embedding_result.size(),
                         token_embedding_result.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    cl_program program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                                    "kernel/elemwise_add.cl");

    kernel_elemwise_add = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_elemwise_add, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(kernel_elemwise_add, 1, sizeof(cl_mem), &bufferPositionalEmbedding);
    err |= clSetKernelArg(kernel_elemwise_add, 2, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSize[] = {token_embedding_result.size()};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_elemwise_add, 1, nullptr, globalSize, nullptr, 0, nullptr,
                                 &event1);

    CHECK_ERROR(err);
//    );


//    util::testBuffer(assetManager, cmdQueue, bufferEmbedding, "encoder/test/positional_embedding_test_fp32.npy");

    // permute
//    PRINT_TIME(2,
    kernel_permute3D_1_0_2 = clCreateKernel(program, "permute3D__1_0_2", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferTemp);
    CHECK_ERROR(err);

    size_t globalSizePermute[3] = {1, CONTEXT_LENGTH, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr, globalSizePermute, nullptr, 1,
                                 &event1,
                                 &event2);
    CHECK_ERROR(err);

//    util::testBuffer(assetManager, cmdQueue, bufferTemp, "encoder/test/permute_test_fp32.npy");
//    );


    // TODO : text_transformer_forward(x)
//    PRINT_TIME(4,
    err = layerNorm0->forward(bufferTemp, bufferEmbedding, 1, &event2, &event3);
    CHECK_ERROR(err);
//    );


    cl_mem bufferAttnInProj0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(float) * token_embedding_result.size() * 3,
                                             nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferAttnInProj0_QKV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              sizeof(float) * token_embedding_result.size() * 3,
                                              nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferAttentionQK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  sizeof(float) * NUM_HEADS * CONTEXT_LENGTH * CONTEXT_LENGTH,
                                                  nullptr, &err);
    CHECK_ERROR(err);

    /* self.model.transformer.resblocks[0].attn.in_proj Linear */
    err = attnInProj0->forward(bufferEmbedding, bufferAttnInProj0, 1, &event3, &event4);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(assetManager, cmdQueue, bufferAttnInProj0, "encoder/test/resblock_0_attn_in_proj_test_fp32.npy");

    /* permute in_proj result to Q,K,V */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    CHECK_ERROR(err);

    size_t globalSizePermute_QKV[3] = {CONTEXT_LENGTH, 3, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr, globalSizePermute_QKV, nullptr, 1,
                                 &event4,
                                 &event5);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(assetManager, cmdQueue, bufferAttnInProj0_QKV, "encoder/test/resblock_0_attn_in_proj_qkv_test_fp32.npy");

    /* permute Q,K,V from (CONTEXT_LENGTH(77), BATCH(1)*NUM_HEADS(16), 1024/NUM_HEADS)
     * to (BATCH(1)*NUM_HEADS(16), CONTEXT_LENGTH(77), 1024/NUM_HEADS) */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferAttnInProj0_QKV);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferAttnInProj0);
    CHECK_ERROR(err);

    size_t head_dim = EMBEDDING_SIZE / NUM_HEADS;
    size_t globalSizePermute_QKV_head[3] = {CONTEXT_LENGTH, NUM_HEADS,head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr, globalSizePermute_QKV_head, nullptr, 1,
                                 &event5,
                                 &event6);

    size_t globalOffsetK[3] = {CONTEXT_LENGTH, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, globalOffsetK, globalSizePermute_QKV_head, nullptr, 1,
                                 &event5,
                                 &event6);

    size_t globalOffsetV[3] = {CONTEXT_LENGTH*2, 0, 0};
    err |= clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, globalOffsetV, globalSizePermute_QKV_head, nullptr, 1,
                                 &event5,
                                 &event6);
    CHECK_ERROR(err);

    // error=0.00001144409179687500
    // util::testBuffer(assetManager, cmdQueue, bufferAttnInProj0, "encoder/test/resblock_0_attn_in_proj_head_test_fp32.npy");

    /* scale dot attention - matmul QxK */
    cl_kernel kernel_add_matmul_attention = clCreateKernel(program, "add_matmul_attention", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_add_matmul_attention, 0, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_add_matmul_attention, 1, sizeof(cl_mem), &bufferAttentionMask);
    err |= clSetKernelArg(kernel_add_matmul_attention, 2, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_add_matmul_attention, 3, sizeof(size_t), &head_dim);
    CHECK_ERROR(err);

    size_t globalSizeQK[3] = {NUM_HEADS, CONTEXT_LENGTH, CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_add_matmul_attention, 3, nullptr, globalSizeQK, nullptr, 3,
                                 &event6,
                                 &event7);

    // util::testBuffer(assetManager, cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_qk_test_fp32.npy");

    /* scale dot attention - softmax */
    cl_kernel kernel_softmax = clCreateKernel(program, "local_softmax", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_softmax, 1, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_softmax, 2, sizeof(float) * CONTEXT_LENGTH, nullptr);
    CHECK_ERROR(err);

    size_t globalSizeSoftmax[1] = {NUM_HEADS * CONTEXT_LENGTH * CONTEXT_LENGTH};
    size_t localSizeSoftmax[1] = {CONTEXT_LENGTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_softmax, 1, nullptr, globalSizeSoftmax, localSizeSoftmax, 1,
                                 &event7,
                                 &event8);
    CHECK_ERROR(err);

    // util::testBuffer(assetManager, cmdQueue, bufferAttentionQK, "encoder/test/resblock_0_attn_softmax_test_fp32.npy");

    /* scale dot attention - attention */
    cl_kernel kernel_matmul_attention = clCreateKernel(program, "batch_matmul_attention", &err);
    CHECK_ERROR(err);

    size_t context_length = CONTEXT_LENGTH;
    err = clSetKernelArg(kernel_matmul_attention, 0, sizeof(cl_mem), &bufferAttentionQK);
    err |= clSetKernelArg(kernel_matmul_attention, 1, sizeof(cl_mem), &bufferAttnInProj0);
    err |= clSetKernelArg(kernel_matmul_attention, 2, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_matmul_attention, 3, sizeof(size_t), &context_length);
    CHECK_ERROR(err);

    size_t globalSizeMatmul[3] = {NUM_HEADS, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_matmul_attention, 3, nullptr, globalSizeMatmul, nullptr, 1,
                                 &event8,
                                 &event9);
    CHECK_ERROR(err);

    // max diff: 0.00000409036874771118
    // util::testBuffer(assetManager, cmdQueue, bufferTemp, "encoder/test/resblock_0_attn_attention_test_fp32.npy");

    /* permute for input of out_proj */
    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSizePermuteOutProj[3] = {NUM_HEADS, CONTEXT_LENGTH, head_dim};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr, globalSizePermuteOutProj, nullptr, 1,
                                 &event9,
                                 &event10);
    CHECK_ERROR(err);

    err = attnOutProj0->forward(bufferEmbedding, bufferTemp, 1, &event10, &event11);
    CHECK_ERROR(err)

    // max diff: 0.00000362098217010498
    // util::testBuffer(assetManager, cmdQueue, bufferTemp, "encoder/test/resblock_0_attn_test_fp32.npy");


    // TODO : x.permute(1, 0, 2)

    // TODO : ln_final(x)

    clWaitForEvents(1, &event11);

    clReleaseProgram(program);
    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferEmbedding);
    clReleaseMemObject(bufferAttnInProj0);
    clReleaseMemObject(bufferAttnInProj0_QKV);
    clReleaseMemObject(bufferAttentionQK);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseEvent(event6);
    clReleaseEvent(event7);
    clReleaseEvent(event8);
    clReleaseEvent(event9);
    clReleaseEvent(event10);
    clReleaseEvent(event11);
    clReleaseKernel(kernel_elemwise_add);
    clReleaseKernel(kernel_permute3D_1_0_2);
    clReleaseKernel(kernel_add_matmul_attention);
    clReleaseKernel(kernel_softmax);
    clReleaseKernel(kernel_matmul_attention);

    auto result = std::vector<float>(token_embedding_result.size());
    return result;
}

void TextEncoder::testEmbedding(const std::vector<long> &token) {
    // test with "a professional photograph of an astronaut riding a horse"
    std::vector<float> token_embedding_result = token_embedding(token);
    auto test = util::load_npy_file(assetManager, "encoder/test/embedding_test_fp32.npy");
    int num = 0;
    float maxDiff = 0;
    for (int i = 0; i < token_embedding_result.size(); i++) {
        if (token_embedding_result[i] != test.data<float>()[i]) {
            num++;
            auto diff = std::abs(token_embedding_result[i] - test.data<float>()[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
    }
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "embedding max diff: %f / num : %d ",
                        maxDiff, num);
}