//
// Created by 구현우 on 2023/12/08.
//

#include "CrossAttention.h"
#include <android/log.h>
#include "../util.h"

#define LOG_TAG "CROSS_ATTENTION"

#define WORK_GROUP_SIZE 64

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

CrossAttention::CrossAttention(
        cl_context context, cl_command_queue cmdQueue, cl_device_id deviceId,
        AAssetManager *assetManager,
        size_t query_dim, size_t context_dim, size_t headSize, size_t headDim,
        const char *q_linear_weight_name,
        const char *k_linear_weight_name,
        const char *v_linear_weight_name,
        const char *out_linear_weight_name, const char *out_linear_bias_name
) : context(context), cmdQueue(cmdQueue), headSize(headSize) {
    cl_int err;

    scale = 1.f / sqrt(static_cast<float>(headDim));

    if (context_dim <= 0) {
        context_dim = query_dim;
    }

    toQLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                           query_dim, headSize * headDim,
                           q_linear_weight_name,
                           nullptr);
    toKLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                           context_dim, headSize * headDim,
                           k_linear_weight_name,
                           nullptr);
    toVLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                           context_dim, headSize * headDim,
                           v_linear_weight_name,
                           nullptr);
    toOutLinear = new Linear(context, cmdQueue, deviceId, assetManager,
                             headSize * headDim, query_dim,
                             out_linear_weight_name,
                             out_linear_bias_name);

    cl_program program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                                    "kernel/util.cl");

    kernel_permute3D_1_0_2 = clCreateKernel(program, "permute3D__1_0_2", &err);
    CHECK_ERROR_THROW(err);

    kernel_softmax = clCreateKernel(program, "softmax", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);

    program = util::create_and_build_program_with_source(context, deviceId, assetManager,
                                                         "kernel/cross_attention.cl");

    kernel_einsum_bik_bjk_bij = clCreateKernel(program, "einsum_bik_bjk_bij", &err);
    CHECK_ERROR_THROW(err);

    kernel_einsum_bij_bjk_bik = clCreateKernel(program, "einsum_bij_bjk_bik", &err);
    CHECK_ERROR_THROW(err);

    clReleaseProgram(program);
}

CrossAttention::~CrossAttention() {
    delete toQLinear;
    delete toKLinear;
    delete toVLinear;
    delete toOutLinear;
    clReleaseKernel(kernel_permute3D_1_0_2);
    clReleaseKernel(kernel_einsum_bik_bjk_bij);
    clReleaseKernel(kernel_einsum_bij_bjk_bik);
    clReleaseKernel(kernel_softmax);
}

cl_int
CrossAttention::forward(cl_mem input, cl_mem condition, cl_mem output, cl_uint num_events_in_list,
                        const cl_event *event_wait_list, cl_event *event) {
    cl_int err;
    cl_event event0_0, event0_1, event0_2;
    cl_event event1_0;
    cl_event event2_0, event2_1, event2_2, event2_3;
    cl_mem bufferQ, bufferK, bufferV, bufferPermuteQ, bufferPermuteK, bufferPermuteV;
    cl_mem bufferEinsumQK, bufferEinsumV, bufferOut;

    if (condition == nullptr) {
        condition = input;
    }

    size_t inputBytes, inputSize, conditionBytes, conditionSize;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    err |= clGetMemObjectInfo(condition, CL_MEM_SIZE, sizeof(size_t), &conditionBytes, nullptr);
    CHECK_ERROR(err);

    inputSize = inputBytes / sizeof(float);
    conditionSize = conditionBytes / sizeof(float);

    bufferQ = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(float) * inputSize / toQLinear->weightShape[1] *
                             toQLinear->weightShape[0],
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(float) * conditionSize / toKLinear->weightShape[1] *
                             toKLinear->weightShape[0],
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(float) * conditionSize / toVLinear->weightShape[1] *
                             toVLinear->weightShape[0],
                             nullptr, &err);
    CHECK_ERROR(err);

    bufferPermuteQ = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * inputSize / toQLinear->weightShape[1] *
                                    toQLinear->weightShape[0],
                                    nullptr, &err);
    CHECK_ERROR(err);

    bufferPermuteK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * conditionSize / toKLinear->weightShape[1] *
                                    toKLinear->weightShape[0],
                                    nullptr, &err);
    CHECK_ERROR(err);

    bufferPermuteV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * conditionSize / toVLinear->weightShape[1] *
                                    toVLinear->weightShape[0],
                                    nullptr, &err);
    CHECK_ERROR(err);

    bufferEinsumQK = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * headSize *
                                    inputSize / toQLinear->weightShape[1] *
                                    conditionSize / toKLinear->weightShape[1],
                                    nullptr, &err);
    CHECK_ERROR(err);

    bufferEinsumV = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * headSize *
                                   inputSize / toQLinear->weightShape[1] *
                                   toVLinear->weightShape[0] / headSize,
                                   nullptr, &err);
    CHECK_ERROR(err);

    bufferOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(float) * headSize *
                               inputSize / toQLinear->weightShape[1] *
                               toVLinear->weightShape[0] / headSize,
                               nullptr, &err);
    CHECK_ERROR(err);


    err = toQLinear->forward(input, bufferQ, num_events_in_list, event_wait_list, &event0_0);
    CHECK_ERROR(err);

    // max diff: 0.00001204013824462891
    // util::testBuffer(cmdQueue, bufferQ, "unet/input_block/test/test_cross_q.npy");

    err = toKLinear->forward(condition, bufferK, num_events_in_list, event_wait_list, &event1_0);
    CHECK_ERROR(err);

    if (cnt == 1) {
        // max diff: 0.00000381469726562500
        // util::testBuffer(cmdQueue, bufferK, "unet/input_block/test/test_basic_attn2_k.npy");
    }

    err = toVLinear->forward(condition, bufferV, num_events_in_list, event_wait_list, &event2_0);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferQ);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferPermuteQ);
    CHECK_ERROR(err);

    /* assume batch size = 1 */
    size_t permuteQGlobalSize[3] = {inputSize / toQLinear->weightShape[1], headSize,
                                    toQLinear->weightShape[0] / headSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 permuteQGlobalSize, nullptr, 1, &event0_0, &event0_1);
    CHECK_ERROR(err);

    // max diff: 0.00001204013824462891
    // util::testBuffer(cmdQueue, bufferPermuteQ, "unet/input_block/test/test_cross_q_permute.npy");

    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferK);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferPermuteK);
    CHECK_ERROR(err);

    size_t permuteKGlobalSize[3] = {conditionSize / toKLinear->weightShape[1], headSize,
                                    toKLinear->weightShape[0] / headSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 permuteKGlobalSize, nullptr, 1, &event1_0, &event0_1);
    CHECK_ERROR(err);

    // max diff: 0.00000947713851928711
    // util::testBuffer(cmdQueue, bufferPermuteK, "unet/input_block/test/test_cross_k_permute.npy");

    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferV);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferPermuteV);
    CHECK_ERROR(err);

    size_t permuteVGlobalSize[3] = {conditionSize / toVLinear->weightShape[1], headSize,
                                    toVLinear->weightShape[0] / headSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 permuteVGlobalSize, nullptr, 1, &event2_0, &event2_1);
    CHECK_ERROR(err);

    size_t kSize = toQLinear->weightShape[0] / headSize;
    err = clSetKernelArg(kernel_einsum_bik_bjk_bij, 0, sizeof(cl_mem), &bufferPermuteQ);
    err |= clSetKernelArg(kernel_einsum_bik_bjk_bij, 1, sizeof(cl_mem), &bufferPermuteK);
    err |= clSetKernelArg(kernel_einsum_bik_bjk_bij, 2, sizeof(cl_mem), &bufferEinsumQK);
    err |= clSetKernelArg(kernel_einsum_bik_bjk_bij, 3, sizeof(size_t), &kSize);
    err |= clSetKernelArg(kernel_einsum_bik_bjk_bij, 4, sizeof(float), &scale);
    CHECK_ERROR(err);

    size_t einsumQKGlobalSize[3] = {headSize, inputSize / toQLinear->weightShape[1],
                                    conditionSize / toKLinear->weightShape[1]};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_einsum_bik_bjk_bij, 3, nullptr,
                                 einsumQKGlobalSize, nullptr, 2, &event0_1, &event0_2);
    CHECK_ERROR(err);

    // max diff: 0.00001931190490722656
    // util::testBuffer(cmdQueue, bufferEinsumQK, "unet/input_block/test/test_cross_einsum.npy");
    // max diff: 0.00003862380981445312
    // util::testBuffer(cmdQueue, bufferEinsumQK, "unet/input_block/test/test_basic_attn2_einsum_qk.npy");

    size_t chunkSize = conditionSize / toKLinear->weightShape[1];
    size_t workGroupSize = WORK_GROUP_SIZE;
//    if (chunkSize % WORK_GROUP_SIZE == 0) {
//        workGroupSize = WORK_GROUP_SIZE;
//    } else {
//        workGroupSize = chunkSize;
//    }
    err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &bufferEinsumQK);
    err |= clSetKernelArg(kernel_softmax, 1, sizeof(cl_mem), &bufferEinsumQK);
    err |= clSetKernelArg(kernel_softmax, 2, sizeof(float) * workGroupSize, nullptr);
    err |= clSetKernelArg(kernel_softmax, 3, sizeof(float) * chunkSize, nullptr);
    err |= clSetKernelArg(kernel_softmax, 4, sizeof(size_t), &chunkSize);
    CHECK_ERROR(err);

    size_t softmaxGlobalSize[1] = {
            headSize * (inputSize / toQLinear->weightShape[1]) * workGroupSize
    };
    size_t softmaxLocalSize[1] = {workGroupSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_softmax, 1, nullptr,
                                 softmaxGlobalSize, softmaxLocalSize, 1, &event0_2, &event2_1);
    CHECK_ERROR(err);

    // max diff: 0.00000052154064178467
    // util::testBuffer(cmdQueue, bufferEinsumQK, "unet/input_block/test/test_cross_softmax.npy");
    // max diff: 0.00000250339508056641
    // util::testBuffer(cmdQueue, bufferEinsumQK, "unet/input_block/test/test_basic_attn2_softmax.npy");

    size_t jSize = conditionSize / toKLinear->weightShape[1];
    err = clSetKernelArg(kernel_einsum_bij_bjk_bik, 0, sizeof(cl_mem), &bufferEinsumQK);
    err |= clSetKernelArg(kernel_einsum_bij_bjk_bik, 1, sizeof(cl_mem), &bufferPermuteV);
    err |= clSetKernelArg(kernel_einsum_bij_bjk_bik, 2, sizeof(cl_mem), &bufferEinsumV);
    err |= clSetKernelArg(kernel_einsum_bij_bjk_bik, 3, sizeof(size_t), &jSize);
    CHECK_ERROR(err);

    size_t einsumVGlobalSize[3] = {headSize, inputSize / toQLinear->weightShape[1],
                                   toVLinear->weightShape[0] / headSize};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_einsum_bij_bjk_bik, 3, nullptr,
                                 einsumVGlobalSize, nullptr, 2, &event2_1, &event2_2);
    CHECK_ERROR(err);

    // max diff: 0.00000193715095520020
    // util::testBuffer(cmdQueue, bufferEinsumV, "unet/input_block/test/test_cross_einsum_v.npy");
    if (cnt == 1) {
        // max diff: 0.00001007318496704102
        // util::testBuffer(cmdQueue, bufferEinsumV,"unet/input_block/test/test_basic_attn2_einsum_v.npy");
    }

    err = clSetKernelArg(kernel_permute3D_1_0_2, 0, sizeof(cl_mem), &bufferEinsumV);
    err |= clSetKernelArg(kernel_permute3D_1_0_2, 1, sizeof(cl_mem), &bufferOut);
    CHECK_ERROR(err);

    size_t permuteOutGlobalSize[3] = {headSize,
                                      (inputSize / toQLinear->weightShape[1]),
                                      (toVLinear->weightShape[0] / headSize)};
    err = clEnqueueNDRangeKernel(cmdQueue, kernel_permute3D_1_0_2, 3, nullptr,
                                 permuteOutGlobalSize, nullptr, 1, &event2_2, &event2_3);
    CHECK_ERROR(err);

    if (cnt == 1) {
        // max diff: 0.00001007318496704102
        // util::testBuffer(cmdQueue, bufferOut, "unet/input_block/test/test_basic_attn2_out.npy");
    }

    err = toOutLinear->forward(bufferOut, output, 1, &event2_3, event);
    CHECK_ERROR(err);

    // max diff: 0.00000205636024475098
    // util::testBuffer(cmdQueue, output, "unet/input_block/test/test_cross_to_out.npy");

    clReleaseEvent(event0_0);
    clReleaseEvent(event0_1);
    clReleaseEvent(event0_2);
    clReleaseEvent(event1_0);
    clReleaseEvent(event2_0);
    clReleaseEvent(event2_1);
    clReleaseEvent(event2_2);
    clReleaseEvent(event2_3);
    clReleaseMemObject(bufferQ);
    clReleaseMemObject(bufferK);
    clReleaseMemObject(bufferV);
    clReleaseMemObject(bufferPermuteQ);
    clReleaseMemObject(bufferPermuteK);
    clReleaseMemObject(bufferPermuteV);
    clReleaseMemObject(bufferEinsumQK);
    clReleaseMemObject(bufferEinsumV);
    clReleaseMemObject(bufferOut);
    cnt += 1;
    return CL_SUCCESS;
}

int CrossAttention::cnt = 0;