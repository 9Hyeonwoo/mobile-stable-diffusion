//
// Created by 구현우 on 2023/12/01.
//

#include "Linear.h"
#include "../util.h"
#include "android/log.h"

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

Linear::Linear(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t in_features, size_t out_features,
        const std::string &weight_name, const std::string &bias_name
) : cmdQueue(cmdQueue), weight_name(weight_name), bias_name(bias_name), event_init_weight(nullptr),
    event_init_bias(nullptr) {
    cl_int err;
    weightShape = std::vector<size_t>({out_features, in_features});

    if (!bias_name.empty()) {
        bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(float) * out_features,
                                    nullptr, &err);
        CHECK_ERROR_THROW(err);
    } else {
        bufferBias = nullptr;
    }

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * out_features * in_features,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/linear.cl");

    kernel = clCreateKernel(program, "linear", &err);
    CHECK_ERROR_THROW(err);

    kernel_reg_linear = clCreateKernel(program, "reg_linear", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

Linear::~Linear() {
    clReleaseKernel(kernel);
    clReleaseKernel(kernel_reg_linear);
    if (event_init_weight != nullptr) {
        clReleaseMemObject(bufferWeight);
        clReleaseEvent(event_init_weight);
    }
    if (event_init_bias != nullptr) {
        clReleaseMemObject(bufferBias);
        clReleaseEvent(event_init_bias);
    }
}

void Linear::init() {
    if (event_init_weight != nullptr && event_init_bias != nullptr) {
        return;
    }
    cl_int err;
    auto weight = util::load_npy_file(weight_name);

    if (weight.num_vals != (weightShape[0] * weightShape[1])) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "weight.num_vals(%ld) != (weightShape[0](%ld) * weightShape[1](%ld))",
                            weight.num_vals, weightShape[0], weightShape[1]);
        throw std::runtime_error("weight.num_vals != (weightShape[0] * weightShape[1])");
    }
    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_TRUE, 0,
                               sizeof(float) * weight.num_vals,
                               weight.data<float>(), 0, nullptr, &event_init_weight);
    CHECK_ERROR_THROW(err);


    if (bufferBias != nullptr) {
        auto bias = util::load_npy_file(bias_name);
        if (bias.num_vals != weightShape[0]) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "bias.num_vals(%ld) != weightShape[0](%ld)",
                                bias.num_vals, weightShape[0]);
            throw std::runtime_error("bias.num_vals != weightShape[0]");
        }

        err = clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_TRUE, 0,
                                   sizeof(float) * bias.num_vals,
                                   bias.data<float>(), 0, nullptr, &event_init_bias);
        CHECK_ERROR_THROW(err);
    }
}

cl_int Linear::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    cl_uint _num_events;
    if (event_init_bias != nullptr) {
        _num_events = num_events_in_list + 2;
    } else {
        _num_events = num_events_in_list + 1;
    }
    auto *_event_list = new cl_event[_num_events];
    _event_list[0] = event_init_weight;
    if (event_init_bias != nullptr) {
        _event_list[1] = event_init_bias;
    }
    for (int i = 0; i < num_events_in_list; i++) {
        _event_list[i + (_num_events - num_events_in_list)] = event_wait_list[i];
    }

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
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, nullptr, globalWorkSize, nullptr,
                                 _num_events, _event_list, event);
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

    err = clSetKernelArg(kernel_reg_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_reg_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_reg_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_reg_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_reg_linear, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel_reg_linear, 5, sizeof(int), &N);
    err |= clSetKernelArg(kernel_reg_linear, 6, sizeof(int), &K);
    err |= clSetKernelArg(kernel_reg_linear, 7, sizeof(size_t), &reg_size_m);
    err |= clSetKernelArg(kernel_reg_linear, 8, sizeof(size_t), &tile_size_m);
    err |= clSetKernelArg(kernel_reg_linear, 9, sizeof(size_t), &tile_size_n);
    err |= clSetKernelArg(kernel_reg_linear, 10, sizeof(float) * tile_size_m * tile_size_k, nullptr);
    err |= clSetKernelArg(kernel_reg_linear, 11, sizeof(float) * tile_size_k * tile_size_n, nullptr);
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
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_reg_linear, 2, nullptr, globalWorkSize_reg_linear,
                                 localWorkSize_reg_linear, _num_events, _event_list, event);
    CHECK_ERROR(err);

    delete[] _event_list;

    return CL_SUCCESS;
}