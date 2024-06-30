//
// Created by 구현우 on 2023/12/01.
//

#include "Linear.h"
#include "../util.h"
#include "android/log.h"
#include "../setting.h"

#define DEBUG 0
#define WIDTH 4
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
        std::shared_ptr<LinearKernel> kernel,
        std::shared_ptr<UtilKernel> utilKernel
) : context(context), cmdQueue(cmdQueue), weight_name(weight_name), bias_name(bias_name),
    bufferWeight(nullptr), bufferBias(nullptr), kernel(kernel), utilKernel(utilKernel) {
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
    if (bufferWeight != nullptr && (bias_name.empty() || bufferBias != nullptr)) {
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
#if LINEAR_KERNEL_VERSION == 0
    /* naive */
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
#elif LINEAR_KERNEL_VERSION == 2
    /** tile(m=11,n=32) without memory copy : light throttle 17295 ms -> 8274ms
     * + local barrier : 4650 ms
     * tile(11, 16) : 5215 ms
     * tile(1, 256) : 21180 ms
     * without tile k but local sync : 8244 ms, 3908 ms, 3905 ms
     * tile_size_k = 1 : 4307 ms, 4046 ms, 3942 ms
     * tile_size_k = 2 : 4422 ms, 4229 ms, 4216 ms
     * tile_size_k = 4 : 4657 ms, 4470 ms, 4486 ms
     * tile_size_k = 8 : 4598 ms
     * tile_size_k = 32 : 4862 ms
     * tile(77, 4) : 27251 ms
     * tile(7, 32) : 5185 ms, 5029 ms, 5003 ms
     * */
    std::vector<size_t> tile_size_ms = {32, 11, 1};
    std::vector<size_t> tile_size_ns = {32};
    int m_index;
    for (m_index = 0; m_index < tile_size_ms.size(); m_index++) {
        if (M % (tile_size_ms[m_index]) == 0) {
            break;
        }
    }

    int n_index;
    for (n_index = 0; n_index < tile_size_ns.size(); n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    if (m_index >= tile_size_ms.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                            __LINE__, M);
        return CL_INVALID_VALUE;
    }
    if (n_index >= tile_size_ns.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                            __LINE__, N);
        return CL_INVALID_VALUE;
    }

    size_t tile_size_m = tile_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];

    err = clSetKernelArg(kernel->tile_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->tile_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->tile_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->tile_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->tile_linear, 4, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N};
    size_t localWorkSize[2] = {tile_size_m, tile_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->tile_linear,
                                 2, nullptr,
                                 globalWorkSize, localWorkSize,
                                 num_events_in_list, event_wait_list, event);
    CHECK_ERROR(err);
#elif LINEAR_KERNEL_VERSION == 3
    /** tile(m=11,n=32) with register n
     * reg_size_m=1 : 4035 ms, 3964 ms, 3924 ms
     * tile(m=11,n=16) reg_size_n=2 : 4295 ms, 4116 ms, 4117 ms
     * reg_size_n=2 : 3712 ms, 3495 ms, 3585 ms
     * local size 단위로 reg reg_size_n=2 : 3776 ms, 3576 ms, 3557 ms
     * reg_size_n=4 : 3852 ms, 3806 ms, 3667 ms
     * tile(m=11,n=64) reg_size_n=4 : 3933 ms, 3562 ms, 3497 ms
     * reg_size_n=8 : 3706 ms, 3620 ms, 3594 ms
     * tile(m=11,n=128) reg_size_n=8 : 3955 ms, 3831 ms, 3856 ms
     * reg_size_n=16 : 9019 ms, 9005 ms, 8924 ms
     * */
    int reg_size_n = 2;
    std::vector<size_t> tile_size_ms = {32, 11, 1};
    std::vector<size_t> tile_size_ns = {32};
    int m_index;
    for (m_index = 0; m_index < tile_size_ms.size(); m_index++) {
        if (M % (tile_size_ms[m_index]) == 0) {
            break;
        }
    }

    int n_index;
    for (n_index = 0; n_index < tile_size_ns.size(); n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    if (m_index >= tile_size_ms.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                            __LINE__, M);
        return CL_INVALID_VALUE;
    }
    if (n_index >= tile_size_ns.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                            __LINE__, N);
        return CL_INVALID_VALUE;
    }

    size_t tile_size_m = tile_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];

    err = clSetKernelArg(kernel->tile_reg_n_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->tile_reg_n_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->tile_reg_n_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->tile_reg_n_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->tile_reg_n_linear, 4, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N / reg_size_n};
    size_t localWorkSize[2] = {tile_size_m, tile_size_n / reg_size_n };
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->tile_reg_n_linear,
                                 2, nullptr,
                                 globalWorkSize, localWorkSize,
                                 num_events_in_list, event_wait_list, event);
    CHECK_ERROR(err);
#elif LINEAR_KERNEL_VERSION == 4
    /** tile(m=11,n=32) with register n and vectorization
     * width=16 + reg_n=2 : 1275 ms, 1265 ms, 1373 ms
     * width=8 : 1306 ms, 1175 ms, 1358 ms
     * width=4 : 1272 ms, 1168 ms, 1233 ms
     * width=2 : 2024 ms, 1954 ms, 1854 ms
     * width=4 + reg_n=8 : 865 ms, 1022 ms, 1145 ms
     * tile(m=11,n=128) width=4 + reg_n=8 : 947 ms, 842 ms, 840 ms
     * */
    /**
     * 개선 여지
     * L/S Unit : Arithmetic Unit = 2 : 1 비율로 확인.
     * full read 거의 못함.
     */
    int reg_size_n = 8;
    std::vector<size_t> tile_size_ms = {32, 11, 1};
    std::vector<size_t> tile_size_ns = {128, 64};
    int m_index;
    for (m_index = 0; m_index < tile_size_ms.size(); m_index++) {
        if (M % (tile_size_ms[m_index]) == 0) {
            break;
        }
    }

    int n_index;
    for (n_index = 0; n_index < tile_size_ns.size(); n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    if (m_index >= tile_size_ms.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                            __LINE__, M);
        return CL_INVALID_VALUE;
    }
    if (n_index >= tile_size_ns.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                            __LINE__, N);
        return CL_INVALID_VALUE;
    }

    size_t tile_size_m = tile_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];

    err = clSetKernelArg(kernel->tile_reg_n_vector_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->tile_reg_n_vector_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->tile_reg_n_vector_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->tile_reg_n_vector_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->tile_reg_n_vector_linear, 4, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {M, N / reg_size_n};
    size_t localWorkSize[2] = {tile_size_m, tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->tile_reg_n_vector_linear,
                                 2, nullptr,
                                 globalWorkSize, localWorkSize,
                                 num_events_in_list, event_wait_list, event);
    CHECK_ERROR(err);
#elif LINEAR_KERNEL_VERSION == 5
    /**
     * tile(m=11,n=128) with register m and n and vectorization
     * reg_size_m=2, reg_size_n=2 : 1.2s 대 나옴. 나아보이는게 없음.
     * reg_size_m=2, reg_size_n=4 : VERSION=4 보다 조금 느림. 100ms 정도. reg_m 이 작아야 좋은건가..
     */
    int reg_size_m = 2;
    int reg_size_n = 4;
    std::vector<size_t> tile_size_ms = {32, 11, 1};
    std::vector<size_t> tile_size_ns = {128};
    int m_index;
    for (m_index = 0; m_index < tile_size_ms.size(); m_index++) {
        if (M % (tile_size_ms[m_index]) == 0) {
            break;
        }
    }

    int n_index;
    for (n_index = 0; n_index < tile_size_ns.size(); n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        }
    }

    if (m_index >= tile_size_ms.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                            __LINE__, M);
        return CL_INVALID_VALUE;
    }
    if (n_index >= tile_size_ns.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                            __LINE__, N);
        return CL_INVALID_VALUE;
    }

    size_t tile_size_m = tile_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];

    auto globalWorkSizeM = M / reg_size_m;
    auto localWorkSizeM = tile_size_m / reg_size_m;
    if (tile_size_m % reg_size_m != 0) {
        localWorkSizeM = tile_size_m / reg_size_m + 1;
        globalWorkSizeM = (M / tile_size_m) * localWorkSizeM;
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "[%s:%d] globalWorkSizeM(%ld) M(%ld) tile_size_m(%ld) %% reg_size_m(%d) != 0\n", __FILE__,
                            __LINE__, globalWorkSizeM, M, tile_size_ms[m_index], reg_size_m);
    }
    if (tile_size_ns[n_index] % reg_size_n != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) tile_size_n(%ld) %% reg_size_n(%d) != 0\n", __FILE__,
                            __LINE__, N, tile_size_ns[n_index], reg_size_n);
        return CL_INVALID_VALUE;
    }


    err = clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 1, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel->tile_reg_m_n_vector_linear, 5, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[2] = {globalWorkSizeM, N / reg_size_n};
    size_t localWorkSize[2] = {localWorkSizeM, tile_size_n / reg_size_n };
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->tile_reg_m_n_vector_linear,
                                 2, nullptr,
                                 globalWorkSize, localWorkSize,
                                 num_events_in_list, event_wait_list, event);
    CHECK_ERROR(err);
#elif LINEAR_KERNEL_VERSION == 6
    /**
     * 2150ms
     * batch matmul 달리 ik kj 방식이 효율적 않았음.
     */
    int reg_size_m = 4;
    // add padding to M
    size_t paddingM = M;
    if (paddingM % reg_size_m != 0) {
        paddingM = (paddingM / reg_size_m + 1) * reg_size_m;
    }

    // permute bufferWeight
    cl_event eventPermute;
    cl_mem bufferWeightPermuted = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(float) * K * N,
                                                 nullptr, &err);
    CHECK_ERROR(err);

    int permuteWeight[3] = {0, 2, 1};
    err = clSetKernelArg(utilKernel->permute3D, 0, sizeof(cl_mem), &bufferWeight);
    err |= clSetKernelArg(utilKernel->permute3D, 1, sizeof(cl_mem), &bufferWeightPermuted);
    err |= clSetKernelArg(utilKernel->permute3D, 2, sizeof(int), &permuteWeight[0]);
    err |= clSetKernelArg(utilKernel->permute3D, 3, sizeof(int), &permuteWeight[1]);
    err |= clSetKernelArg(utilKernel->permute3D, 4, sizeof(int), &permuteWeight[2]);
    CHECK_ERROR(err);

    size_t globalWorkSizePermute[3] = {1, N, K};
    err = clEnqueueNDRangeKernel(
            cmdQueue, utilKernel->permute3D,
            3, nullptr, globalWorkSizePermute, nullptr,
            num_events_in_list, event_wait_list, &eventPermute
    );
    CHECK_ERROR(err);

    std::vector<size_t> tile_size_ms = {40};
    std::vector<size_t> tile_size_ns = {128};
    int m_index;
    for (m_index = 0; m_index < tile_size_ms.size(); m_index++) {
        if (paddingM % (tile_size_ms[m_index]) == 0) {
            break;
        } else if (m_index == tile_size_ms.size() - 1) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "[%s:%d] M(%ld) %% tile_size_m != 0\n", __FILE__,
                                __LINE__, M);
            return CL_INVALID_VALUE;
        }
    }

    int n_index;
    for (n_index = 0; n_index < tile_size_ns.size(); n_index++) {
        if (N % (tile_size_ns[n_index]) == 0) {
            break;
        } else if (n_index == tile_size_ns.size() - 1) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "[%s:%d] N(%ld) %% tile_size_n != 0\n", __FILE__,
                                __LINE__, N);
            return CL_INVALID_VALUE;
        }
    }

    size_t tile_size_m = tile_size_ms[m_index];
    size_t tile_size_n = tile_size_ns[n_index];

    auto globalWorkSizeM = paddingM / reg_size_m;
    auto localWorkSizeM = tile_size_m / reg_size_m;
    if (tile_size_m % reg_size_m != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "[%s:%d] globalWorkSizeM(%ld) M(%ld) tile_size_m(%ld) %% reg_size_m(%d) != 0\n",
                            __FILE__,
                            __LINE__, globalWorkSizeM, M, tile_size_ms[m_index], reg_size_m);
        return CL_INVALID_VALUE;
    }
    if (tile_size_n % WIDTH != 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "[%s:%d] N(%ld) tile_size_n(%ld) %% WIDTH(%d) != 0\n", __FILE__,
                            __LINE__, N, tile_size_ns[n_index], WIDTH);
        return CL_INVALID_VALUE;
    }

    err = clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 1, sizeof(cl_mem), &bufferWeightPermuted);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 2, sizeof(cl_mem), &bufferBias);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 5, sizeof(int), &N);
    err |= clSetKernelArg(kernel->tile_reg_m_vector_n_linear, 6, sizeof(int), &K);
    CHECK_ERROR(err);

    size_t globalWorkSize[3] = {1, globalWorkSizeM, N / WIDTH};
    size_t localWorkSize[3] = {1, localWorkSizeM, tile_size_n / WIDTH};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->tile_reg_m_vector_n_linear,
                                 3, nullptr,
                                 globalWorkSize, localWorkSize,
                                 1, &eventPermute, event);
    CHECK_ERROR(err);

    clReleaseMemObject(bufferWeightPermuted);
    clReleaseEvent(eventPermute);
#elif LINEAR_KERNEL_VERSION == 1
    /* register : light throttle 15818 ms -> 8319 ms */
    size_t reg_size_n = 8;
    size_t tile_size_k = 16;
    cl_uchar tile_size_ms[] = {128, 77, 1};
    cl_uchar reg_size_ms[] = {8, 7, 1};
    cl_uchar tile_size_ns[] = {128, 64};

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
    cl_uchar tile_size_m = tile_size_ms[m_index];
    cl_uchar reg_size_m = reg_size_ms[m_index];
    cl_uchar tile_size_n = tile_size_ns[n_index];
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
    err |= clSetKernelArg(kernel->register_linear, 7, sizeof(cl_uchar), &reg_size_m);
    err |= clSetKernelArg(kernel->register_linear, 8, sizeof(cl_uchar), &tile_size_m);
    err |= clSetKernelArg(kernel->register_linear, 9, sizeof(cl_uchar), &tile_size_n);
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
    size_t localWorkSize_reg_linear[2] = {static_cast<size_t>(tile_size_m / reg_size_m), tile_size_n / reg_size_n};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel->register_linear, 2, nullptr,
                                 globalWorkSize_reg_linear,
                                 localWorkSize_reg_linear, num_events_in_list, event_wait_list,
                                 event);
    CHECK_ERROR(err);
#endif

#if DEBUG
    clWaitForEvents(1, event);
    if (count == 0)
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "try, component, index, input size, out feature, in feature, tile size m, tile size n, reg size n, kernel, time(ms)\n");
    auto message =
            "0, Linear, " +
            std::to_string(count++) + ", " +
            std::to_string(inputSize) + ", " +
            std::to_string(weightShape[0]) + ", " +
            std::to_string(weightShape[1]) + ", " +
            std::to_string(tile_size_m) + ", " +
            std::to_string(tile_size_n) + ", " +
#if LINEAR_KERNEL_VERSION == 6
            std::to_string(WIDTH);
#else
            std::to_string(reg_size_n);
#endif
    util::printEventTime(message + ", register_linear", *event);
#endif

    return CL_SUCCESS;
}