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

    bufferPositionalEmbedding = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       positional_embedding.num_bytes(),
                                       nullptr, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferPositionalEmbedding, CL_TRUE, 0,
                         positional_embedding.num_bytes(),
                         positional_embedding.data<float>(), 0, nullptr, nullptr);

    CHECK_ERROR(err);

    block0 = new ResidualAttentionBlock(context, cmdQueue, deviceId, assetManager, NUM_HEADS);
}

TextEncoder::~TextEncoder() {
    delete block0;
    clReleaseMemObject(bufferPositionalEmbedding);
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
    cl_event event1, event2, event3;
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
    err = block0->forward(bufferTemp, bufferEmbedding, 1, &event2, &event3);
    CHECK_ERROR(err);
    // TODO : x.permute(1, 0, 2)

    // TODO : ln_final(x)

    clWaitForEvents(1, &event3);

    clReleaseProgram(program);
    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferEmbedding);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseKernel(kernel_elemwise_add);
    clReleaseKernel(kernel_permute3D_1_0_2);

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