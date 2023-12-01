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

std::shared_ptr<_cl_program> Linear::program = nullptr;

Linear::Linear(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
               AAssetManager *assetManager, const char *weight_name, const char *bias_name)
               : context(context), cmdQueue(cmdQueue) {
    cl_int err;
    auto weight = util::load_npy_file(assetManager, weight_name);
    auto bias = util::load_npy_file(assetManager, bias_name);
    weightShape = weight.shape;
    biasShape = bias.shape;
    if (weight.shape[0] != bias.shape[0]) {
        throw std::runtime_error("weight.shape[0] != bias.shape[0]");
    }


    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * weight.num_vals,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(float) * bias.num_vals,
                                nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_TRUE, 0,
                               sizeof(float) * weight.num_vals,
                               weight.data<float>(), 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_TRUE, 0,
                                sizeof(float) * bias.num_vals,
                                bias.data<float>(), 0, nullptr, nullptr);
    CHECK_ERROR_THROW(err);

    if (!program || program.use_count() == 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "create program");
        program = std::shared_ptr<_cl_program>(
                util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                           "kernel/linear.cl"),
                [](_cl_program *p) { clReleaseProgram(p); }
        );
    }
}

Linear::~Linear() {
    clReleaseMemObject(bufferWeight);
    clReleaseMemObject(bufferBias);
}

cl_int Linear::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    auto inputSize = inputBytes / sizeof(float);
    if (inputSize % weightShape[1] != 0) {
        throw std::runtime_error("inputSize % weightShape[1] != 0");
    }

    cl_kernel kernel = clCreateKernel(program.get(), "linear", &err);
    CHECK_ERROR(err);

    auto M = inputSize / weightShape[1];
    auto N = weightShape[0];
    auto K = weightShape[1];
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, nullptr, globalWorkSize, nullptr, num_events_in_list,
                                 event_wait_list, event);
    CHECK_ERROR(err);

    clReleaseKernel(kernel);

    return CL_SUCCESS;
}