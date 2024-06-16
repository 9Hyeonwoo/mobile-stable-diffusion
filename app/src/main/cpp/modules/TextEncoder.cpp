//
// Created by 구현우 on 2023/11/24.
//

#include "TextEncoder.h"
#include "util.h"

#include <chrono>

#define LOG_TAG "TEXT_ENCODER"
#define EMBEDDING_SIZE 1024
#define NUM_HEADS 16
#define CONTEXT_LENGTH 77
#define LAYERS (24-1)

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

#define PRINT_TIME(index, ...) \
    do { \
        auto start##index = std::chrono::steady_clock::now(); \
        { __VA_ARGS__; } \
        auto end##index = std::chrono::steady_clock::now(); \
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "PRINT_TIME[%s]: %0.3f", #index, std::chrono::duration_cast<std::chrono::microseconds>(end##index - start##index).count() / 1000.0); \
    } while(0)

TextEncoder::TextEncoder(
        AAssetManager *assetManager,
        cl_context context,
        cl_command_queue cmdQueue,
        cl_device_id deviceId
) : context(context), cmdQueue(cmdQueue) {
    embedding = util::load_npy_file("encoder/embedding_fp32.npy");
    bufferPositionalEmbedding = util::load_npy_file("encoder/positional_embedding_fp32.npy",
                                                    nullptr, context, cmdQueue);
    bufferAttentionMask = util::load_npy_file("encoder/attn_mask_fp32.npy", nullptr, context,
                                              cmdQueue);

    layerNormKernel = std::make_shared<LayerNormKernel>(context, deviceId, assetManager);
    linearKernel = std::make_shared<LinearKernel>(context, deviceId, assetManager);
    multiHeadAttentionKernel = std::make_shared<MultiHeadAttentionKernel>(context, deviceId,
                                                                          assetManager);
    utilKernel = std::make_shared<UtilKernel>(context, deviceId, assetManager);

    for (int i = 0; i < LAYERS; i++) {
        auto folder_prefix =
                "encoder/resblock/" + std::to_string(i) + "/resblock_" + std::to_string(i);
        auto ln_1_weight_name = folder_prefix + "_ln_1_weight_fp32.npy";
        auto ln_1_bias_name = folder_prefix + "_ln_1_bias_fp32.npy";
        auto ln_2_weight_name = folder_prefix + "_ln_2_weight_fp32.npy";
        auto ln_2_bias_name = folder_prefix + "_ln_2_bias_fp32.npy";
        auto attn_in_proj_weight_name = folder_prefix + "_attn_in_proj_weight_fp32.npy";
        auto attn_in_proj_bias_name = folder_prefix + "_attn_in_proj_bias_fp32.npy";
        auto attn_out_proj_weight_name = folder_prefix + "_attn_out_proj_weight_fp32.npy";
        auto attn_out_proj_bias_name = folder_prefix + "_attn_out_proj_bias_fp32.npy";
        auto mlp_c_fc_weight_name = folder_prefix + "_mlp_c_fc_weight_fp32.npy";
        auto mlp_c_fc_bias_name = folder_prefix + "_mlp_c_fc_bias_fp32.npy";
        auto mlp_c_proj_weight_name = folder_prefix + "_mlp_c_proj_weight_fp32.npy";
        auto mlp_c_proj_bias_name = folder_prefix + "_mlp_c_proj_bias_fp32.npy";
        resBlocks.push_back(
                new ResidualAttentionBlock(context, cmdQueue,
                                           EMBEDDING_SIZE, NUM_HEADS,
                                           ln_1_weight_name, ln_1_bias_name,
                                           ln_2_weight_name, ln_2_bias_name,
                                           attn_in_proj_weight_name,
                                           attn_in_proj_bias_name,
                                           attn_out_proj_weight_name,
                                           attn_out_proj_bias_name,
                                           mlp_c_fc_weight_name, mlp_c_fc_bias_name,
                                           mlp_c_proj_weight_name,
                                           mlp_c_proj_bias_name,
                                           bufferAttentionMask,
                                           layerNormKernel,
                                           linearKernel,
                                           multiHeadAttentionKernel,
                                           utilKernel)
        );
        resBlocks[i]->init();
    }

    ln_final = new LayerNorm(context, cmdQueue,
                             EMBEDDING_SIZE,
                             "encoder/ln_final_weight_fp32.npy",
                             "encoder/ln_final_bias_fp32.npy",
                             layerNormKernel);
    ln_final->init();
}

TextEncoder::~TextEncoder() {
    for (auto block: resBlocks) {
        delete block;
    }
    delete ln_final;
    clReleaseMemObject(bufferPositionalEmbedding);
    clReleaseMemObject(bufferAttentionMask);
}

/*
 * @input: `token` tokenized text
 * @return: token embedding with size=(length of 'token'(CONTEXT_LENGTH=77) * (EMBEDDING_SIZE=1024))
 */
cl_mem TextEncoder::createTokenEmbeddingBuffer(const std::vector<long> &token) {
    cl_int err;
    auto buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(float) * token.size() * EMBEDDING_SIZE,
                                 nullptr, &err);
    CHECK_ERROR(err)

    auto data = clEnqueueMapBuffer(cmdQueue, buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                            sizeof(float) * token.size() * EMBEDDING_SIZE, 0, nullptr, nullptr,
                                            &err);
    CHECK_ERROR(err)

    for (auto i = 0; i < token.size(); i++) {
        auto tokenEmbedding = embedding.data<float>() + (token[i] * EMBEDDING_SIZE);
        std::copy(tokenEmbedding, tokenEmbedding + EMBEDDING_SIZE, static_cast<float *>(data) + i * EMBEDDING_SIZE);
    }

    clEnqueueUnmapMemObject(cmdQueue, buffer, data, 0, nullptr, nullptr);

    return buffer;
}

std::vector<float> TextEncoder::encode(const std::vector<long> &token) {
    cl_int err;
    cl_event event1, event2, event3, event4, event5, event6;
    cl_mem bufferEmbedding, bufferTemp;

    // elemwise_add
    bufferEmbedding = createTokenEmbeddingBuffer(token);
    // util::testBuffer(cmdQueue, bufferEmbedding, "encoder/test/embedding_test_fp32.npy");

    bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * token.size() * EMBEDDING_SIZE,
                                       nullptr, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(utilKernel->elemwise_add, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(utilKernel->elemwise_add, 1, sizeof(cl_mem), &bufferPositionalEmbedding);
    err |= clSetKernelArg(utilKernel->elemwise_add, 2, sizeof(cl_mem), &bufferEmbedding);
    CHECK_ERROR(err);

    size_t globalSize[] = {token.size() * EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->elemwise_add, 1, nullptr, globalSize, nullptr,
                                 0,
                                 nullptr,
                                 &event1);
    CHECK_ERROR(err);


//    util::testBuffer(cmdQueue, bufferEmbedding, "encoder/test/positional_embedding_test_fp32.npy");

    // permute
//    PRINT_TIME(2,

    err = clSetKernelArg(utilKernel->permute3D_1_0_2, 0, sizeof(cl_mem), &bufferEmbedding);
    err |= clSetKernelArg(utilKernel->permute3D_1_0_2, 1, sizeof(cl_mem), &bufferTemp);
    CHECK_ERROR(err);

    size_t globalSizePermute[3] = {1, CONTEXT_LENGTH, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_1_0_2, 3, nullptr,
                                 globalSizePermute,
                                 nullptr, 1,
                                 &event1,
                                 &event2);
    CHECK_ERROR(err);

//    util::testBuffer(cmdQueue, bufferTemp, "encoder/test/permute_test_fp32.npy");
//    );

    /* text_transformer_forward(x) */
    // init swapped state
    auto &inBuffer = bufferEmbedding;
    auto &outBuffer = bufferTemp;
    auto &inEvent = event3;
    auto &outEvent = event2;
    for (auto block: resBlocks) {
        // swap buffer and event
        std::swap(inBuffer, outBuffer);
        std::swap(inEvent, outEvent);

        err = block->forward(inBuffer, outBuffer, 1, &inEvent, &outEvent);
        CHECK_ERROR(err);
    }

    // max diff: 0.00003051757812500000
    // util::testBuffer(cmdQueue, outBuffer, "encoder/test/resblock_22_test_fp32.npy");

    /* x.permute(1, 0, 2) */
    err = clSetKernelArg(utilKernel->permute3D_1_0_2, 0, sizeof(cl_mem), &outBuffer);
    err |= clSetKernelArg(utilKernel->permute3D_1_0_2, 1, sizeof(cl_mem), &inBuffer);
    CHECK_ERROR(err);

    size_t globalSizePermuteReverse[3] = {CONTEXT_LENGTH, 1, EMBEDDING_SIZE};
    err = clEnqueueNDRangeKernel(cmdQueue, utilKernel->permute3D_1_0_2, 3, nullptr,
                                 globalSizePermuteReverse,
                                 nullptr, 1,
                                 &outEvent,
                                 &event4);
    CHECK_ERROR(err);

    /* ln_final(x) */
    err = ln_final->forward(inBuffer, outBuffer, 1, &event4, &event5);
    CHECK_ERROR(err);

    // max diff: 0.00002861022949218750
     util::testBuffer(cmdQueue, outBuffer, "encoder/test/ln_final_test_fp32.npy");
    auto result = std::vector<float>(token.size() * EMBEDDING_SIZE);
    err = clEnqueueReadBuffer(cmdQueue, outBuffer, CL_TRUE, 0,
                              sizeof(float) * result.size(),
                              result.data(), 1, &event5, &event6);

    clWaitForEvents(1, &event6);

    clReleaseMemObject(bufferTemp);
    clReleaseMemObject(bufferEmbedding);
    clReleaseEvent(event1);
    clReleaseEvent(event2);
    clReleaseEvent(event3);
    clReleaseEvent(event4);
    clReleaseEvent(event5);
    clReleaseEvent(event6);

    return result;
}