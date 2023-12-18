//
// Created by 구현우 on 2023/12/07.
//

#include "Conv2D.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "CONV2D"

#define WORK_GROUP_SIZE (64 * 64)

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

Conv2D::Conv2D(
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t in_channel, size_t out_channel, size_t kernel_size, int stride, int padding,
        const std::string &weight_name,
        const std::string &bias_name
) : context(context), cmdQueue(cmdQueue), stride(stride), padding(padding),
    weight_name(weight_name), bias_name(bias_name),
    event_init_weight(nullptr), event_init_bias(nullptr) {
    cl_int err;

    weightShape = std::vector<size_t>({out_channel, in_channel, kernel_size, kernel_size});
    biasShape = std::vector<size_t>({out_channel});

    bufferWeight = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * out_channel * in_channel * kernel_size *
                                  kernel_size,
                                  nullptr, &err);
    CHECK_ERROR_THROW(err);

    bufferBias = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(float) * out_channel,
                                nullptr, &err);
    CHECK_ERROR_THROW(err);

    auto program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                              "kernel/conv2d.cl");

    /*
    kernel = clCreateKernel(program, "conv2d", &err);
    CHECK_ERROR_THROW(err);

    kernel_im2col = clCreateKernel(program, "im2col", &err);
    CHECK_ERROR_THROW(err);

    kernel_conv2d_matmul = clCreateKernel(program, "conv2d_matmul", &err);
    CHECK_ERROR_THROW(err);
    */

    kernel_im2win = clCreateKernel(program, "im2win", &err);
    CHECK_ERROR_THROW(err);

    kernel_im2win_matmul = clCreateKernel(program, "im2win_matmul", &err);
    CHECK_ERROR_THROW(err);

    kernel_im2win_batch_matmul = clCreateKernel(program, "im2win_batch_matmul", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

Conv2D::~Conv2D() {
    /*
    clReleaseKernel(kernel);
    clReleaseKernel(kernel_im2col);
    clReleaseKernel(kernel_conv2d_matmul);
    */
    clReleaseKernel(kernel_im2win);
    clReleaseKernel(kernel_im2win_matmul);
    clReleaseKernel(kernel_im2win_batch_matmul);
    if (event_init_weight != nullptr) {
        clReleaseMemObject(bufferWeight);
        clReleaseEvent(event_init_weight);
    }
    if (event_init_bias != nullptr) {
        clReleaseMemObject(bufferBias);
        clReleaseEvent(event_init_bias);
    }
}

void Conv2D::init() {
    if (event_init_weight != nullptr && event_init_bias != nullptr) {
        return;
    }
    cl_int err;
    auto weight = util::load_npy_file(weight_name);
    auto bias = util::load_npy_file(bias_name);

    if (weight.num_vals != (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D weight file size != constructor weightShape");
        throw std::runtime_error("Conv2D weight file size != constructor weightShape");
    }

    if (bias.num_vals != biasShape[0]) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D bias file size != constructor biasShape");
        throw std::runtime_error("Conv2D bias file size != constructor biasShape");
    }

    err = clEnqueueWriteBuffer(cmdQueue, bufferWeight, CL_TRUE, 0,
                               sizeof(float) * weight.num_vals,
                               weight.data<float>(), 0, nullptr, &event_init_weight);
    CHECK_ERROR_THROW(err);

    err = clEnqueueWriteBuffer(cmdQueue, bufferBias, CL_TRUE, 0,
                               sizeof(float) * bias.num_vals,
                               bias.data<float>(), 0, nullptr, &event_init_bias);
    CHECK_ERROR_THROW(err);
}

/*
 * Assume square shaped `input` where height = width.
 */
cl_int Conv2D::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    cl_mem bufferCol, bufferWin;
    cl_event _event[1];
    auto _num_event_list = num_events_in_list + 2;
    auto *event_list = new cl_event[_num_event_list];
    event_list[0] = event_init_weight;
    event_list[1] = event_init_bias;
    for (int i = 0; i < num_events_in_list; i++) {
        event_list[i + 2] = event_wait_list[i];
    }

    if (input == output) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Conv2D not support input == output");
        throw std::runtime_error("Conv2D not support input == output");
    }

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    auto inputSize = inputBytes / sizeof(float);
    inputSize /= weightShape[1];
    inputSize = static_cast<size_t>(sqrt(static_cast<float>(inputSize)));

    if (inputSize * inputSize * weightShape[1] != inputBytes / sizeof(float)) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D inputSize * inputSize * weightShape[1] != inputBytes / sizeof(float)");
        throw std::runtime_error(
                "Conv2D inputSize * inputSize * weightShape[1] != inputBytes / sizeof(float)");
    }

    auto outputSize = getOutputSize(inputSize);

    /* naive */
    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel, 5, sizeof(size_t), &weightShape[1]);
    err |= clSetKernelArg(kernel, 6, sizeof(size_t), &weightShape[2]);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &padding);
    CHECK_ERROR(err);

    size_t globalSize[3] = {weightShape[0], outputSize, outputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, nullptr, globalSize, nullptr,
                                 _num_event_list, event_list, event);
    CHECK_ERROR(err);
    */
    /* naive */

    /* im2col version */
    /*
    size_t kernel_size = weightShape[2];
    size_t in_channel = weightShape[1];
    bufferCol = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(float) * (in_channel * kernel_size * kernel_size) *
                               (outputSize * outputSize),
                               nullptr, &err);
    CHECK_ERROR(err);

    size_t num_kernels = in_channel * outputSize * outputSize;
    int im_offset = 0;
    int col_offset = 0;
    err = clSetKernelArg(kernel_im2col, 0, sizeof(size_t), &num_kernels);
    err |= clSetKernelArg(kernel_im2col, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_im2col, 2, sizeof(int), &im_offset);
    err |= clSetKernelArg(kernel_im2col, 3, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel_im2col, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel_im2col, 5, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel_im2col, 6, sizeof(int), &padding);
    err |= clSetKernelArg(kernel_im2col, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel_im2col, 8, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2col, 9, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2col, 10, sizeof(cl_mem), &bufferCol);
    err |= clSetKernelArg(kernel_im2col, 11, sizeof(int), &col_offset);
    CHECK_ERROR(err);

    size_t globalSize_im2col[1] = {num_kernels};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_im2col, 1, nullptr, globalSize_im2col, nullptr,
                                 _num_event_list, event_list, &_event[0]);
    CHECK_ERROR(err);

    size_t out_channel = weightShape[0];
    size_t N = outputSize * outputSize;
    size_t K = in_channel * kernel_size * kernel_size;

    err = clSetKernelArg(kernel_conv2d_matmul, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_conv2d_matmul, 1, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_conv2d_matmul, 2, sizeof(cl_mem), &bufferCol);
    err |= clSetKernelArg(kernel_conv2d_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_conv2d_matmul, 4, sizeof(size_t), &out_channel);
    err |= clSetKernelArg(kernel_conv2d_matmul, 5, sizeof(size_t), &N);
    err |= clSetKernelArg(kernel_conv2d_matmul, 6, sizeof(size_t), &K);
    CHECK_ERROR(err);

    size_t globalSize_conv2d_matmul[1] = {out_channel * N};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_conv2d_matmul, 1, nullptr,
                                 globalSize_conv2d_matmul, nullptr,
                                 1, &_event[0], event);
    CHECK_ERROR(err);

    clReleaseMemObject(bufferCol);
    */
    /* im2col version */

    /* im2win version */
    size_t kernel_size = weightShape[2];
    size_t in_channel = weightShape[1];
    size_t width_pad = (inputSize + 2 * padding);
    bufferWin = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(float) * (in_channel * outputSize) *
                               (width_pad * kernel_size),
                               nullptr, &err);
    CHECK_ERROR(err);

    int im_offset = 0;
    int col_offset = 0;
    size_t num_windows = in_channel * outputSize * width_pad;
    size_t width_win = width_pad * kernel_size;
    err = clSetKernelArg(kernel_im2win, 0, sizeof(size_t), &num_windows);
    err |= clSetKernelArg(kernel_im2win, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_im2win, 2, sizeof(int), &im_offset);
    err |= clSetKernelArg(kernel_im2win, 3, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel_im2win, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel_im2win, 5, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel_im2win, 6, sizeof(int), &padding);
    err |= clSetKernelArg(kernel_im2win, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel_im2win, 8, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2win, 9, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel_im2win, 10, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel_im2win, 11, sizeof(int), &col_offset);
    CHECK_ERROR(err);

    size_t globalSize_im2win[1] = {num_windows};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_im2win, 1, nullptr, globalSize_im2win, nullptr,
                                 _num_event_list, event_list, &_event[0]);
    CHECK_ERROR(err);

    size_t out_channel = weightShape[0];

    /* im2win matmul - naive
    err = clSetKernelArg(kernel_im2win_matmul, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_im2win_matmul, 1, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_im2win_matmul, 2, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel_im2win_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_im2win_matmul, 4, sizeof(size_t), &out_channel);
    err |= clSetKernelArg(kernel_im2win_matmul, 5, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2win_matmul, 6, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2win_matmul, 7, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel_im2win_matmul, 8, sizeof(size_t), &in_channel);
    err |= clSetKernelArg(kernel_im2win_matmul, 9, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel_im2win_matmul, 10, sizeof(int), &stride);
    CHECK_ERROR(err);

    size_t globalSize_im2win_matmul[1] = {out_channel * outputSize * outputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_im2win_matmul, 1, nullptr,
                                 globalSize_im2win_matmul, nullptr,
                                 1, &_event[0], event);
    CHECK_ERROR(err);
    im2win matmul - naive */

    size_t tile_size_m = 1, reg_size_n = 8;
    size_t tile_size_ns[] = {128, 64, 32, 16, 8};
    size_t tile_size_ks[] = {16, 4};

    int n_index, n_size = 5;
    for (n_index = 0; n_index < n_size; n_index++) {
        if (outputSize % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    int k_index, k_size = 2;
    for (k_index = 0; k_index < k_size; k_index++) {
        if (in_channel % (tile_size_ks[k_index]) == 0) {
            break;
        }
    }

    if (n_index >= n_size) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] outputSize(%ld) %% tile_size_n(%ld) != 0\n", __FILE__,
                            __LINE__, outputSize, tile_size_ns[0]);
        return CL_INVALID_VALUE;
    }
    if (k_index >= n_size) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] in_channel(%ld) %% tile_size_k(%ld) != 0\n", __FILE__,
                            __LINE__, in_channel, tile_size_ks[0]);
        return CL_INVALID_VALUE;
    }
    size_t tile_size_n = tile_size_ns[n_index];
    size_t tile_size_k = tile_size_ks[k_index];
    err = clSetKernelArg(kernel_im2win_batch_matmul, 0, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 4, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 5, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 6, sizeof(size_t), &in_channel);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 7, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 8, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 9, sizeof(int), &stride);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 10,
                          sizeof(float) * tile_size_k * tile_size_m *
                          (kernel_size * kernel_size + (tile_size_n - 1) * stride * kernel_size),
                          nullptr);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 11,
                          sizeof(float) * tile_size_k * kernel_size * kernel_size, nullptr);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 12, sizeof(size_t), &tile_size_n);
    err |= clSetKernelArg(kernel_im2win_batch_matmul, 13, sizeof(size_t), &tile_size_k);
    CHECK_ERROR(err);

    size_t globalSize_im2win_batch_matmul[3] = {out_channel, outputSize, outputSize / reg_size_n};
    size_t localSize_im2win_batch_matmul[3] = {1, 1, tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_im2win_batch_matmul, 3, nullptr,
                                 globalSize_im2win_batch_matmul, localSize_im2win_batch_matmul,
                                 1, &_event[0], event);
    CHECK_ERROR(err);

    clReleaseMemObject(bufferWin);
    /* im2win version */


    for (auto &e: _event) {
        clReleaseEvent(e);
    }

    delete[] event_list;

    return CL_SUCCESS;
}

size_t Conv2D::getOutputSize(size_t inputSize) {
    return (inputSize + 2 * padding - weightShape[2]) / stride + 1;
}