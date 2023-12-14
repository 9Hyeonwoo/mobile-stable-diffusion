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
        event_init_bias = nullptr;
    }

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * out_features * in_features,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/linear.cl");

    kernel = clCreateKernel(program, "linear", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

Linear::~Linear() {
    clReleaseMemObject(bufferWeight);
    clReleaseMemObject(bufferBias);
    clReleaseKernel(kernel);
    clReleaseEvent(event_init_weight);
    if (event_init_bias != nullptr) {
        clReleaseEvent(event_init_bias);
    }
}

void Linear::init() {
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

    delete[] _event_list;

    return CL_SUCCESS;
}