//
// Created by 구현우 on 2023/11/24.
//

#include "TextEncoder.h"
#include "util.h"

#include <chrono>

#define LOG_TAG "TEXT_ENCODER"
#define EMBEDDING_SIZE 1024

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

cnpy::NpyArray *load_npy_file(AAssetManager *assetManager, const char *filename) {
    AAsset *asset = AAssetManager_open(assetManager, filename, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        throw std::runtime_error("Failed to open the asset.");
    }

    auto buffer = static_cast<const unsigned char *>(AAsset_getBuffer(asset));
    auto length = AAsset_getLength(asset);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(buffer, word_size, shape, fortran_order);

    auto arr = new cnpy::NpyArray(shape, word_size, fortran_order);
    size_t offset = length - arr->num_bytes();
    memcpy(arr->data<char>(), buffer + offset, arr->num_bytes());

    AAsset_close(asset);
    return arr;
}

TextEncoder::TextEncoder(AAssetManager *assetManager, cl_context context, cl_command_queue cmdQueue,
                         cl_device_id deviceId) : context(context), cmdQueue(cmdQueue),
                                                  deviceId(deviceId), assetManager(assetManager) {
    embedding = load_npy_file(assetManager, "encoder/embedding_fp32.npy");
    positional_embedding = load_npy_file(assetManager, "encoder/positional_embedding_fp32.npy");
}

TextEncoder::~TextEncoder() {
    delete embedding;
    delete positional_embedding;
}

/*
 * @input: `token` tokenized text
 * @return: token embedding with size=(length of 'token'(CONTEXT_LENGTH=77) * (EMBEDDING_SIZE=1024))
 */
std::vector<float> TextEncoder::token_embedding(const std::vector<long> &token) {
    std::vector<float> result;
    for (auto i: token) {
        auto data = embedding->data<float>() + (i * EMBEDDING_SIZE);
        result.insert(result.end(), data, data + EMBEDDING_SIZE);
    }
    return result;
}

std::vector<float> TextEncoder::encode(const std::vector<long> &token) {
    cl_int err;
//    PRINT_TIME(0,
    std::vector<float> token_embedding_result = token_embedding(token);

    // elemwise_add
    cl_mem bufferEmbedding = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(float) * token_embedding_result.size(),
                                            nullptr, nullptr);
    cl_mem bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       positional_embedding->num_bytes(),
                                       nullptr, nullptr);

    clEnqueueWriteBuffer(cmdQueue, bufferEmbedding, CL_FALSE, 0,
                         sizeof(float) * token_embedding_result.size(),
                         token_embedding_result.data(), 0, nullptr, nullptr);

    clEnqueueWriteBuffer(cmdQueue, bufferTemp, CL_FALSE, 0,
                         positional_embedding->num_bytes(),
                         positional_embedding->data<float>(), 0, nullptr, nullptr);

    cl_program program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                                    "kernel/elemwise_add.cl");

    cl_kernel kernel = clCreateKernel(program, "elemwise_add", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSize[] = {token_embedding_result.size()};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, globalSize, nullptr, 0, nullptr,
                                 nullptr);

    CHECK_ERROR(err);
//    );

    clReleaseKernel(kernel);


    std::vector<float> result(token_embedding_result.size());
    clEnqueueReadBuffer(cmdQueue, bufferEmbedding, CL_TRUE, 0,
                        sizeof(float) * token_embedding_result.size(),
                        result.data(), 0, nullptr, nullptr);


//    PRINT_TIME(1,
    for (int i = 0; i < token_embedding_result.size(); i++) {
        token_embedding_result[i] += positional_embedding->data<float>()[i];
        if (result[i] != token_embedding_result[i]) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "elemwise_add error: %f, %f",
                                result[i], token_embedding_result[i]);
            throw std::runtime_error("elemwise_add error.");
        }
    }
//    );

    // permute
//    PRINT_TIME(2,
    kernel = clCreateKernel(program, "permute3D__1_0_2", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferTemp);
    CHECK_ERROR(err);

    size_t globalSizePermute[3] = {1, 77, 1024};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, nullptr, globalSizePermute, nullptr, 0,
                                 nullptr,
                                 nullptr);
    CHECK_ERROR(err);
//    );


    clEnqueueReadBuffer(cmdQueue, bufferTemp, CL_TRUE, 0,
                        sizeof(float) * token_embedding_result.size(),
                        result.data(), 0, nullptr, nullptr);
//    PRINT_TIME(3,
    int shape[3] = {1, 77, 1024};
    int dimensions[3] = {1, 0, 2};
    auto permute_result = util::permute3D(token_embedding_result, shape, dimensions);
    for (int i = 0; i < permute_result->size(); i++) {
        if (result[i] != permute_result->at(i)) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "permute error: %f != %f",
                                result[i], permute_result->at(i));
            throw std::runtime_error("permute error.");
        }
    }
    delete permute_result;
//    );

    // TODO : text_transformer_forward(x)

    // TODO : x.permute(1, 0, 2)

    // TODO : ln_final(x)

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferEmbedding);
    return result;
}