//
// Created by 구현우 on 2023/12/08.
//

#include "BasicTransformerBlock.h"

#include <android/log.h>
#include "../util.h"

#define LOG_TAG "BASIC_TRANSFORMER_BLOCK"

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

BasicTransformerBlock::BasicTransformerBlock(
        cl_context context, cl_command_queue cmdQueue,
        size_t dim, size_t context_dim, size_t headSize, size_t headDim,
        const std::string &layer_norm_1_weight_name, const std::string &layer_norm_1_bias_name,
        const std::string &layer_norm_2_weight_name, const std::string &layer_norm_2_bias_name,
        const std::string &layer_norm_3_weight_name, const std::string &layer_norm_3_bias_name,
        const std::string &cross_1_q_linear_weight_name,
        const std::string &cross_1_k_linear_weight_name,
        const std::string &cross_1_v_linear_weight_name,
        const std::string &cross_1_out_linear_weight_name, const std::string &cross_1_out_linear_bias_name,
        const std::string &cross_2_q_linear_weight_name,
        const std::string &cross_2_k_linear_weight_name,
        const std::string &cross_2_v_linear_weight_name,
        const std::string &cross_2_out_linear_weight_name, const std::string &cross_2_out_linear_bias_name,
        const std::string &ff_geglu_linear_weight_name, const std::string &ff_geglu_linear_bias_name,
        const std::string &ff_net_linear_weight_name, const std::string &ff_net_linear_bias_name,
        std::shared_ptr<LayerNormKernel> layerNormKernel,
        std::shared_ptr<LinearKernel> linearKernel,
        std::shared_ptr<UtilKernel> utilKernel,
        std::shared_ptr<CrossAttentionKernel> crossAttentionKernel,
        std::shared_ptr<GEGLUKernel> gegluKernel
) : cmdQueue(cmdQueue), context(context), utilKernel(utilKernel) {

    layerNorm1 = new LayerNorm(context, cmdQueue, dim,
                               layer_norm_1_weight_name, layer_norm_1_bias_name, layerNormKernel);
    layerNorm2 = new LayerNorm(context, cmdQueue, dim,
                               layer_norm_2_weight_name, layer_norm_2_bias_name, layerNormKernel);
    layerNorm3 = new LayerNorm(context, cmdQueue, dim,
                               layer_norm_3_weight_name, layer_norm_3_bias_name, layerNormKernel);
    crossAttention1 = new CrossAttention(context, cmdQueue,
                                         dim, 0, headSize, headDim,
                                         cross_1_q_linear_weight_name,
                                         cross_1_k_linear_weight_name,
                                         cross_1_v_linear_weight_name,
                                         cross_1_out_linear_weight_name,
                                         cross_1_out_linear_bias_name,
                                         linearKernel,
                                         utilKernel,
                                         crossAttentionKernel);
    crossAttention2 = new CrossAttention(context, cmdQueue,
                                         dim, context_dim, headSize, headDim,
                                         cross_2_q_linear_weight_name,
                                         cross_2_k_linear_weight_name,
                                         cross_2_v_linear_weight_name,
                                         cross_2_out_linear_weight_name,
                                         cross_2_out_linear_bias_name,
                                         linearKernel,
                                         utilKernel,
                                         crossAttentionKernel);
    feedForward = new FeedForward(context, cmdQueue, dim,
                                  ff_geglu_linear_weight_name, ff_geglu_linear_bias_name,
                                  ff_net_linear_weight_name, ff_net_linear_bias_name,
                                  linearKernel, gegluKernel, utilKernel);
}

void BasicTransformerBlock::init() {
    layerNorm1->init();
    layerNorm2->init();
    layerNorm3->init();
    crossAttention1->init();
    crossAttention2->init();
    feedForward->init();
}

BasicTransformerBlock::~BasicTransformerBlock() {
    delete layerNorm1;
    delete layerNorm2;
    delete layerNorm3;
    delete crossAttention1;
    delete crossAttention2;
    delete feedForward;
}

cl_int BasicTransformerBlock::forward(cl_mem input, cl_mem condition, cl_mem output,
                                      cl_uint num_events_in_list, const cl_event *event_wait_list,
                                      cl_event *event) {
    cl_int err;
    cl_event event0, event1, event2, event3, event4, event5, event6, event7;
    cl_mem bufferNorm, bufferNorm2;

    size_t inputBytes;
    err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &inputBytes, nullptr);
    CHECK_ERROR(err);

    size_t inputSize = inputBytes / sizeof(float);

    bufferNorm = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBytes, nullptr, &err);
    CHECK_ERROR(err);

    bufferNorm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, inputBytes, nullptr, &err);
    CHECK_ERROR(err);

    err = layerNorm1->forward(input, bufferNorm, num_events_in_list, event_wait_list, &event0);
    CHECK_ERROR(err);

    // max diff: 0.00000542029738426208
    // util::testBuffer(cmdQueue, bufferNorm, "unet/input_block/test/test_basic_norm1.npy");

    err = crossAttention1->forward(bufferNorm, nullptr, bufferNorm,
                                   1, &event0, &event1);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferNorm);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &bufferNorm);
    CHECK_ERROR(err);

    size_t globalSize[1] = {inputSize};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr, globalSize, nullptr,
                                 1, &event1, &event2);
    CHECK_ERROR(err);

    err = layerNorm2->forward(bufferNorm, bufferNorm2, 1, &event2, &event3);
    CHECK_ERROR(err);

    // max diff: 0.00000345706939697266
    // util::testBuffer(cmdQueue, bufferNorm2, "unet/input_block/test/test_basic_norm2.npy");

    err = crossAttention2->forward(bufferNorm2, condition, bufferNorm2,
                                   1, &event3, &event4);
    CHECK_ERROR(err);

    // max diff: 0.00000175833702087402
    // util::testBuffer(cmdQueue, bufferNorm2, "unet/input_block/test/test_basic_attn2.npy");

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferNorm2);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &bufferNorm);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &bufferNorm2);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr, globalSize, nullptr,
                                 1, &event4, &event5);
    CHECK_ERROR(err);

    err = layerNorm3->forward(bufferNorm2, bufferNorm, 1, &event5, &event6);
    CHECK_ERROR(err);

    // max diff: 0.00000476837158203125
    // util::testBuffer(cmdQueue, bufferNorm, "unet/input_block/test/test_basic_norm_3.npy");

    err = feedForward->forward(bufferNorm, bufferNorm, 1, &event6, &event7);
    CHECK_ERROR(err);

    // max diff: 0.00000560283660888672
    // util::testBuffer(cmdQueue, bufferNorm, "unet/input_block/test/test_basic_ff.npy");

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferNorm);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &bufferNorm2);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr, globalSize, nullptr,
                                 1, &event7, event);
    CHECK_ERROR(err);

    // max diff: 0.00000572204589843750
    // util::testBuffer(cmdQueue, output, "unet/input_block/test/test_basic.npy");

    clReleaseEvent(event0);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseEvent(event6);
    clReleaseEvent(event7);
    clReleaseMemObject(bufferNorm);
    clReleaseMemObject(bufferNorm2);

    return CL_SUCCESS;
}