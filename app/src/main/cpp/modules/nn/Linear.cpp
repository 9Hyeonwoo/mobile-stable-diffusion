//
// Created by 구현우 on 2023/12/01.
//

#include "Linear.h"
#include "../util.h"
#include "android/log.h"

#define DEBUG 0
#define LOG_TAG "LINEAR"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      return err; \
    }

#define CHECK_ERROR_THROW(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

static int count = 0;

Linear::Linear(
        cl_context context, cl_command_queue cmdQueue,
        size_t in_features, size_t out_features,
        const std::string &weight_name, const std::string &bias_name,
        std::shared_ptr<LinearKernel> kernel
) : context(context), cmdQueue(cmdQueue), weight_name(weight_name), bias_name(bias_name),
    bufferWeight(nullptr), bufferBias(nullptr), kernel(kernel) {
    weightShape = std::vector<size_t>({out_features, in_features});
}

Linear::~Linear() {
    if (bufferWeight != nullptr) {
        clReleaseMemObject(bufferWeight);
    }
    if (bufferBias != nullptr) {
        clReleaseMemObject(bufferBias);
    }
}

void Linear::init() {
    if (bufferWeight != nullptr && bufferBias != nullptr) {
        return;
    }
    size_t weight_num_vals;
    bufferWeight = util::load_npy_file(weight_name, &weight_num_vals, context, cmdQueue);

    if (weight_num_vals != (weightShape[0] * weightShape[1])) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "weight.num_vals(%ld) != (weightShape[0](%ld) * weightShape[1](%ld))",
                            weight_num_vals, weightShape[0], weightShape[1]);
        throw std::runtime_error("weight.num_vals != (weightShape[0] * weightShape[1])");
    }

    if (!bias_name.empty()) {
        size_t bias_num_vals;
        bufferBias = util::load_npy_file(bias_name, &bias_num_vals, context, cmdQueue);
        if (bias_num_vals != weightShape[0]) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "bias.num_vals(%ld) != weightShape[0](%ld)",
                                bias_num_vals, weightShape[0]);
            throw std::runtime_error("bias.num_vals != weightShape[0]");
        }
    }
}

cl_int Linear::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;

    if (input == output) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Linear not support input == output");
        throw std::runtime_error("Linear not support input == output");
    }

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    auto inputSize = inputBytes / sizeof(float);
    if (inputSize % weightShape[1] != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] inputSize(%ld) %% weightShape[1](%ld) != 0\n", __FILE__,
                            __LINE__, inputSize, weightShape[1]);
        throw std::runtime_error("inputSize % weightShape[1] != 0");
    }

    auto M = inputSize / weightShape[1];
    auto N = weightShape[0];
    auto K = weightShape[1];
    /* naive
    err = clSetKernelArg(kernel->naive_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->naive_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->naive_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->naive_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->naive_linear, 4, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->naive_linear, 2, nullptr, globalWorkSize, nullptr,
                                 num_events_in_list, event_wait_list, event);
    CHECK_ERROR(err);
     naive */

    size_t reg_size_n = 8;
    size_t tile_size_k = 16;
    size_t tile_size_ms[] = {128, 77, 1};
    size_t reg_size_ms[] = {8, 7, 1};
    size_t tile_size_ns[] = {128, 64};

    int m_index, m_size = 3;
    for (m_index = 0; m_index < m_size; m_index++) {
        if (M % (tile_size_ms[m_index]) == 0) {
            break;
        }
    }

    int n_index, n_size = 2;
    for (n_index = 0; n_index < n_size; n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    if (m_index >= m_size) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                            __LINE__, M);
        return CL_INVALID_VALUE;
    }
    if (n_index >= n_size) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                            __LINE__, N);
        return CL_INVALID_VALUE;
    }
    size_t tile_size_m = tile_size_ms[m_index];
    size_t reg_size_m = reg_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];
    if (K % tile_size_k != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] K(%ld) %% tile_size_k(%ld) != 0\n", __FILE__,
                            __LINE__, K, tile_size_k);
        return CL_INVALID_VALUE;
    }

    err = clSetKernelArg(kernel->register_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->register_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->register_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->register_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->register_linear, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel->register_linear, 5, sizeof(int), &N);
    err |= clSetKernelArg(kernel->register_linear, 6, sizeof(int), &K);
    err |= clSetKernelArg(kernel->register_linear, 7, sizeof(size_t), &reg_size_m);
    err |= clSetKernelArg(kernel->register_linear, 8, sizeof(size_t), &tile_size_m);
    err |= clSetKernelArg(kernel->register_linear, 9, sizeof(size_t), &tile_size_n);
    err |= clSetKernelArg(kernel->register_linear, 10, sizeof(float) * tile_size_m * tile_size_k,
                          nullptr);
    err |= clSetKernelArg(kernel->register_linear, 11, sizeof(float) * tile_size_k * tile_size_n,
                          nullptr);
    CHECK_ERROR(err);

    size_t globalSize_m, globalSize_n;
    if (M % (tile_size_m) != 0) {
        globalSize_m = ((M / tile_size_m) + 1) * tile_size_m;
    } else {
        globalSize_m = M;
    }
    if (N % (tile_size_n) != 0) {
        globalSize_n = ((N / tile_size_n) + 1) * tile_size_n;
    } else {
        globalSize_n = N;
    }
    size_t globalWorkSize_reg_linear[2] = {globalSize_m / reg_size_m, globalSize_n / reg_size_n};
    size_t localWorkSize_reg_linear[2] = {tile_size_m / reg_size_m, tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->register_linear, 2, nullptr,
                                 globalWorkSize_reg_linear,
                                 localWorkSize_reg_linear, num_events_in_list, event_wait_list,
                                 event);
    CHECK_ERROR(err);

#if DEBUG
    clWaitForEvents(1, event);
    if (count == 0)
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "try, component, index, input size, out feature, in feature, tile size m, tile size n, tile size k, reg size m, reg size n, kernel, time(ms)\n");
    auto message =
            "0, Linear, " +
            std::to_string(count++) + ", " +
            std::to_string(inputSize) + ", " +
            std::to_string(weightShape[0]) + ", " +
            std::to_string(weightShape[1]) +  ", " +
            std::to_string(tile_size_m) + ", " +
            std::to_string(tile_size_n) + ", " +
            std::to_string(tile_size_k) + ", " +
            std::to_string(reg_size_m) + ", " +
            std::to_string(reg_size_n);
    util::printEventTime(message + ", register_linear", *event);
#endif

    return CL_SUCCESS;
}