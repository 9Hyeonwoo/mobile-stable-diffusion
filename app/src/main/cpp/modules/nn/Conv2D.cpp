//
// Created by 구현우 on 2023/12/07.
//

#include "Conv2D.h"
#include <android/log.h>
#include "../util.h"
#include "../setting.h"

#define DEBUG 0
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

static int count = 0;

Conv2D::Conv2D(
        cl_context context,
        cl_command_queue cmdQueue,
        size_t in_channel, size_t out_channel, size_t kernel_size, int stride, int padding,
        const std::string &weight_name,
        const std::string &bias_name,
        std::shared_ptr<ConvKernel> kernel
) : context(context), cmdQueue(cmdQueue), stride(stride), padding(padding),
    weight_name(weight_name), bias_name(bias_name),
    bufferWeight(nullptr), bufferBias(nullptr), kernel(kernel) {

    weightShape = std::vector<size_t>({out_channel, in_channel, kernel_size, kernel_size});
    biasShape = std::vector<size_t>({out_channel});
}

Conv2D::~Conv2D() {
    if (bufferWeight != nullptr) {
        clReleaseMemObject(bufferWeight);
    }
    if (bufferWeight != nullptr) {
        clReleaseMemObject(bufferBias);
    }
}

void Conv2D::init() {
    if (bufferWeight != nullptr && bufferBias != nullptr) {
        return;
    }
    size_t weight_num_vals, bias_num_vals;

    bufferWeight = util::load_npy_file(weight_name, &weight_num_vals, context, cmdQueue);
    bufferBias = util::load_npy_file(bias_name, &bias_num_vals, context, cmdQueue);

    if (weight_num_vals != (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D weight file size != constructor weightShape");
        throw std::runtime_error("Conv2D weight file size != constructor weightShape");
    }

    if (bias_num_vals != biasShape[0]) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Conv2D bias file size != constructor biasShape");
        throw std::runtime_error("Conv2D bias file size != constructor biasShape");
    }
}

/*
 * Assume square shaped `input` where height = width.
 */
cl_int Conv2D::forward(cl_mem input, cl_mem output, cl_uint num_events_in_list,
                       const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    cl_mem bufferCol, bufferWin;
    cl_event _event[1];

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
    err = clSetKernelArg(kernel->conv2d, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->conv2d, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->conv2d, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->conv2d, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->conv2d, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel->conv2d, 5, sizeof(size_t), &weightShape[1]);
    err |= clSetKernelArg(kernel->conv2d, 6, sizeof(size_t), &weightShape[2]);
    err |= clSetKernelArg(kernel->conv2d, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel->conv2d, 8, sizeof(int), &padding);
    CHECK_ERROR(err);

    size_t globalSize[3] = {weightShape[0], outputSize, outputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->conv2d, 3, nullptr, globalSize, nullptr,
                                 num_events_in_list, event_wait_list, event);
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
    err = clSetKernelArg(kernel->im2col, 0, sizeof(size_t), &num_kernels);
    err |= clSetKernelArg(kernel->im2col, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->im2col, 2, sizeof(int), &im_offset);
    err |= clSetKernelArg(kernel->im2col, 3, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel->im2col, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel->im2col, 5, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel->im2col, 6, sizeof(int), &padding);
    err |= clSetKernelArg(kernel->im2col, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel->im2col, 8, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2col, 9, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2col, 10, sizeof(cl_mem), &bufferCol);
    err |= clSetKernelArg(kernel->im2col, 11, sizeof(int), &col_offset);
    CHECK_ERROR(err);

    size_t globalSize_im2col[1] = {num_kernels};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->im2col, 1, nullptr, globalSize_im2col, nullptr,
                                 num_events_in_list, event_wait_list, &_event[0]);
    CHECK_ERROR(err);

    size_t out_channel = weightShape[0];
    size_t N = outputSize * outputSize;
    size_t K = in_channel * kernel_size * kernel_size;

    err = clSetKernelArg(kernel->conv2d_matmul, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->conv2d_matmul, 1, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->conv2d_matmul, 2, sizeof(cl_mem), &bufferCol);
    err |= clSetKernelArg(kernel->conv2d_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->conv2d_matmul, 4, sizeof(size_t), &out_channel);
    err |= clSetKernelArg(kernel->conv2d_matmul, 5, sizeof(size_t), &N);
    err |= clSetKernelArg(kernel->conv2d_matmul, 6, sizeof(size_t), &K);
    CHECK_ERROR(err);

    size_t globalSize_conv2d_matmul[1] = {out_channel * N};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->conv2d_matmul, 1, nullptr,
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
    err = clSetKernelArg(kernel->im2win, 0, sizeof(size_t), &num_windows);
    err |= clSetKernelArg(kernel->im2win, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->im2win, 2, sizeof(int), &im_offset);
    err |= clSetKernelArg(kernel->im2win, 3, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel->im2win, 4, sizeof(size_t), &inputSize);
    err |= clSetKernelArg(kernel->im2win, 5, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel->im2win, 6, sizeof(int), &padding);
    err |= clSetKernelArg(kernel->im2win, 7, sizeof(int), &stride);
    err |= clSetKernelArg(kernel->im2win, 8, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2win, 9, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel->im2win, 10, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel->im2win, 11, sizeof(int), &col_offset);
    CHECK_ERROR(err);

    size_t globalSize_im2win[1] = {num_windows};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->im2win, 1, nullptr, globalSize_im2win, nullptr,
                                 num_events_in_list, event_wait_list, &_event[0]);
    CHECK_ERROR(err);

    size_t out_channel = weightShape[0];

    /* im2win matmul - naive */
#if CONV_2D_KERNEL_VERSION == 0
    err = clSetKernelArg(kernel->im2win_matmul, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->im2win_matmul, 1, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->im2win_matmul, 2, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel->im2win_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->im2win_matmul, 4, sizeof(size_t), &out_channel);
    err |= clSetKernelArg(kernel->im2win_matmul, 5, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2win_matmul, 6, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2win_matmul, 7, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel->im2win_matmul, 8, sizeof(size_t), &in_channel);
    err |= clSetKernelArg(kernel->im2win_matmul, 9, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel->im2win_matmul, 10, sizeof(int), &stride);
    CHECK_ERROR(err);

    size_t globalSize_im2win_matmul[1] = {out_channel * outputSize * outputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->im2win_matmul, 1, nullptr,
                                 globalSize_im2win_matmul, nullptr,
                                 1, &_event[0], event);
    CHECK_ERROR(err);
    /* im2win matmul - naive */
#elif CONV_2D_KERNEL_VERSION == 1
    size_t tile_size_n = 256;
    size_t reg_size_n = 16;
    size_t MN = outputSize * outputSize;

    if (MN % tile_size_n != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] MN(%ld) %% tile_size_n(%ld) != 0\n", __FILE__,
                            __LINE__, MN, tile_size_n);
        return CL_INVALID_VALUE;
    }

    err = clSetKernelArg(kernel->im2win_reg_n_matmul, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 1, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 2, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 4, sizeof(int), &outputSize);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 5, sizeof(int), &outputSize);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 6, sizeof(int), &width_win);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 7, sizeof(int), &in_channel);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 8, sizeof(int), &kernel_size);
    err |= clSetKernelArg(kernel->im2win_reg_n_matmul, 9, sizeof(int), &stride);
    CHECK_ERROR(err);

    size_t globalSize_im2win_matmul[2] = {out_channel,  MN / reg_size_n};
    size_t localSize_im2win_matmul[2] = {1, tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->im2win_reg_n_matmul, 2, nullptr,
                                 globalSize_im2win_matmul, localSize_im2win_matmul,
                                 1, &_event[0], event);
    CHECK_ERROR(err);
#endif

    /* im2win matmul - register
    size_t tile_size_m = 1, reg_size_n = 4;
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
    if (k_index >= k_size) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] in_channel(%ld) %% tile_size_k(%ld) != 0\n", __FILE__,
                            __LINE__, in_channel, tile_size_ks[0]);
        return CL_INVALID_VALUE;
    }
    size_t tile_size_n = tile_size_ns[n_index];
    size_t tile_size_k = tile_size_ks[k_index];
    err = clSetKernelArg(kernel->im2win_batch_matmul, 0, sizeof(cl_mem), &bufferWin);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 4, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 5, sizeof(size_t), &outputSize);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 6, sizeof(size_t), &in_channel);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 7, sizeof(size_t), &width_win);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 8, sizeof(size_t), &kernel_size);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 9, sizeof(int), &stride);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 10,
                          sizeof(float) * tile_size_k * tile_size_m *
                          (kernel_size * kernel_size + (tile_size_n - 1) * stride * kernel_size),
                          nullptr);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 11,
                          sizeof(float) * tile_size_k * kernel_size * kernel_size, nullptr);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 12, sizeof(size_t), &tile_size_n);
    err |= clSetKernelArg(kernel->im2win_batch_matmul, 13, sizeof(size_t), &tile_size_k);
    CHECK_ERROR(err);

    size_t globalSize_im2win_batch_matmul[3] = {out_channel, outputSize, outputSize / reg_size_n};
    size_t localSize_im2win_batch_matmul[3] = {1, 1, tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->im2win_batch_matmul, 3, nullptr,
                                 globalSize_im2win_batch_matmul, localSize_im2win_batch_matmul,
                                 1, &_event[0], event);
    CHECK_ERROR(err);
     im2win matmul - register */

#if DEBUG
    clWaitForEvents(1, event);
    if (count == 0)
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "try, component, index, input size, output size, in channel, out channel, kernel size, kernel, time(ms)\n");
    auto message =
            "0, Conv2D, " +
            std::to_string(count++) + ", " +
            std::to_string(inputSize) + ", " +
            std::to_string(outputSize) + ", " +
            std::to_string(weightShape[1]) + ", " +
            std::to_string(weightShape[0]) + ", " +
            std::to_string(weightShape[2]);
    util::printEventTime(message + ", im2win", _event[0]);
    util::printEventTime(message + ", im2win_matmul", *event);
#endif

    clReleaseMemObject(bufferWin);
    /* im2win version */


    for (auto &e: _event) {
        clReleaseEvent(e);
    }

    return CL_SUCCESS;
}

size_t Conv2D::getOutputSize(size_t inputSize) {
    return (inputSize + 2 * padding - weightShape[2]) / stride + 1;
}