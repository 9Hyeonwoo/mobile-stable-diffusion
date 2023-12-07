//
// Created by 구현우 on 2023/12/07.
//

#include "UNetModel.h"

#include "util.h"

#define MODEL_CHANNELS 320

/*
 * Assume Batch size 'B' is 1.
 * @param x: [B, LATENT_CHANNEL(4), HEIGHT/DOWN_SAMPLING(512/8=64), WIDTH/DOWN_SAMPLING(512/8=64)]
 * @param timestep: long. originally [B]
 * @param condition: [B, CONTEXT_LENGTH(77), EMBEDDING_SIZE(1024)]
 */
std::vector<float> UNetModel::forward(const std::vector<float> &x, long timestep,
                                      const std::vector<float> &condition) {
    return std::vector<float>();
}

/*
 * @param timestep: long. originally [B]
 * @return [B, MODEL_CHANNELS(320)]
 */
std::vector<float> UNetModel::timestep_embedding(long timestep) {
    float max_period = 10000.0f;
    int half = MODEL_CHANNELS / 2;

    std::vector<float> embedding(MODEL_CHANNELS);
    for (int i = 0; i < half; i++) {
        auto freq = exp((-log(max_period)) * static_cast<float>(i) / static_cast<float>(half));
        auto arg = static_cast<float>(timestep) * freq;
        embedding[i] = cos(arg);
        embedding[i + half] = sin(arg);
    }

    // timestep=981. max diff: 0.00000005960464477539
    // util::testBuffer(embedding, "unet/test/test_timestep_embedding.npy");
    return embedding;
}