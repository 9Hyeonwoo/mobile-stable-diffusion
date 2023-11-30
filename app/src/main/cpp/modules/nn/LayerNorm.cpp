//
// Created by 구현우 on 2023/11/27.
//

#include "LayerNorm.h"
#include "../util.h"
#include <android/log.h>

#define LOG_TAG "LAYER_NORM"
#define MAX_WORK_GROUP_SIZE 512

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

std::shared_ptr<_cl_program> LayerNorm::program = nullptr;

LayerNorm::LayerNorm(cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
                     AAssetManager *assetManager, const char *weight_name, const char *bias_name)
        : context(context), cmdQueue(cmdQueue), assetManager(assetManager) {
    cl_int err;
    auto weight = util::load_npy_file(assetManager, weight_name);
    auto bias = util::load_npy_file(assetManager, bias_name);
    weightSize = weight->num_vals;
    biasSize = bias->num_vals;
    if (weightSize != biasSize) {
        throw std::runtime_error("weightSize != biasSize");
    }

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * weightSize,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(float) * biasSize,
                                nullptr, &err);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_FALSE, 0, sizeof(float) * weightSize,
                               weight->data<float>(), 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_FALSE, 0, sizeof(float) * biasSize,
                                bias->data<float>(), 0, nullptr, nullptr);
    CHECK_ERROR_THROW(err);

    if (!program || program.use_count() == 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "create program");
        program = std::shared_ptr<_cl_program>(
                util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                           "kernel/layer_norm.cl"),
                [](_cl_program *p) { clReleaseProgram(p); }
        );
    }
}

LayerNorm::~LayerNorm() {
    clReleaseMemObject(bufferWeight);
    clReleaseMemObject(bufferBias);
}

cl_int LayerNorm::forward(cl_mem input, cl_mem output) {
    cl_int err;
    size_t input_bytes;
    cl_event event1, event2;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &input_bytes, nullptr);
    CHECK_ERROR(err);

    auto input_size = input_bytes / sizeof(cl_float);

    if (input_size % weightSize != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "input_size: %ld, weight->num_vals: %ld",
                            input_size, weightSize);
        throw std::runtime_error("input_size % weight->num_vals != 0");
    }

    cl_mem bufferMean = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * input_size / weightSize,
                                       nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferVariance = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(float) * input_size / weightSize,
                                           nullptr, &err);
    CHECK_ERROR(err);

    cl_mem bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * input_size / MAX_WORK_GROUP_SIZE,
                                       nullptr, &err);

    cl_kernel kernel_mean = clCreateKernel(program.get(), "local_mean", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_mean, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_mean, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_mean, 2, sizeof(size_t), &weightSize);
    err |= clSetKernelArg(kernel_mean, 3, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_mean, 4, sizeof(float) * MAX_WORK_GROUP_SIZE, nullptr);
    CHECK_ERROR(err);

    size_t globalWorkSize[1] = {input_size};
    size_t localWorkSize[1] = {MAX_WORK_GROUP_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_mean, 1, nullptr, globalWorkSize, localWorkSize,
                                 0, nullptr, &event1);
    CHECK_ERROR(err);

    clFinish(cmdQueue);
    util::testBuffer(assetManager, cmdQueue, bufferMean, "encoder/test/local_mean_test_fp32.npy");
    clFinish(cmdQueue);

    cl_kernel kernel_var = clCreateKernel(program.get(), "local_variance", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_var, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_var, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_var, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel_var, 3, sizeof(size_t), &weightSize);
    err |= clSetKernelArg(kernel_var, 4, sizeof(cl_mem), &bufferTemp);
    err |= clSetKernelArg(kernel_var, 5, sizeof(float) * weightSize, nullptr);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel_var, 1, nullptr, globalWorkSize, localWorkSize, 1,
                                 &event1, &event2);
    CHECK_ERROR(err);

    cl_kernel kernel_norm = clCreateKernel(program.get(), "layer_norm", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_norm, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_norm, 1, sizeof(cl_mem), &bufferMean);
    err |= clSetKernelArg(kernel_norm, 2, sizeof(cl_mem), &bufferVariance);
    err |= clSetKernelArg(kernel_norm, 3, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_norm, 4, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_norm, 5, sizeof(size_t), &weightSize);
    err |= clSetKernelArg(kernel_norm, 6, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, kernel_norm, 1, nullptr, globalWorkSize, nullptr, 1,
                                 &event2, nullptr);
    CHECK_ERROR(err);

    clReleaseKernel(kernel_mean);
    clReleaseKernel(kernel_var);
    clReleaseKernel(kernel_norm);
    clReleaseMemObject(bufferMean);
    clReleaseMemObject(bufferVariance);
    clReleaseMemObject(bufferTemp);
    clReleaseEvent(event1);
    clReleaseEvent(event2);

    return CL_SUCCESS;
}