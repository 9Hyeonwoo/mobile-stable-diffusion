//
// Created by 구현우 on 2023/12/05.
//

#include "DDIMSampler.h"
#include <algorithm>
#include <numeric>
#include <random>
#include "util.h"
#include <android/log.h>

#define LOG_TAG "DDIM_SAMPLER"
#define DDPM_NUM_TIME_STEPS 1000
#define SEED 42
#define LINEAR_START 0.00085
#define LINEAR_END 0.0120

DDIMSampler::DDIMSampler(const std::function<std::vector<float>(
        const std::vector<float> &, int, const std::vector<float> &)> &apply_model
) : apply_model(apply_model) {
    std::vector<double> betas = linspace(sqrt(LINEAR_START), sqrt(LINEAR_END), DDPM_NUM_TIME_STEPS);
    for (auto &beta: betas) {
        beta = beta * beta;
    }

    std::vector<double> alphas;
    for (auto &beta: betas) {
        alphas.push_back(1.0f - beta);
    }

    auto d_alphas_cumprod = std::vector<double>(alphas.size());
    std::partial_sum(alphas.begin(), alphas.end(), d_alphas_cumprod.begin(),
                     [](auto acc, auto element) {
                         return acc * element;
                     });
    alphas_cumprod = std::vector<float>(d_alphas_cumprod.begin(), d_alphas_cumprod.end());
    // correct.
    // util::testBuffer(alphas_cumprod, "sampler/test/test_alphas_cumprod.npy");
}

DDIMSampler::~DDIMSampler() = default;

std::vector<float> DDIMSampler::sample(
        std::vector<float> *x_T,
        int ddim_num_steps,
        const int shape[3],
        const std::vector<float> &conditioning) {
    std::vector<float> img;
    if (x_T == nullptr) {
        std::mt19937 gen(SEED);
        std::normal_distribution<float> normalDist(0.0f, 1.0f);
        img = std::vector<float>(shape[0] * shape[1] * shape[2]);
        for (float &i: img) {
            i = normalDist(gen);
        }
    } else {
        img = *x_T;
    }

    std::vector<int> ddim_timesteps;
    int c = DDPM_NUM_TIME_STEPS / ddim_num_steps;
    for (int i = 0; i < DDPM_NUM_TIME_STEPS; i += c) {
        ddim_timesteps.push_back(i + 1);
    }

    auto schedule = make_schedule(ddim_timesteps);
    auto alphas = schedule.first;
    auto alphas_prev = schedule.second;
    std::vector<float> sqrt_one_minus_alphas(alphas.size());
    for (int i = 0; i < alphas.size(); i++) {
        sqrt_one_minus_alphas[i] = sqrt(1.0f - alphas[i]);
    }
    // correct.
    // util::testBuffer(alphas, "sampler/test/test_alphas.npy");
    // util::testBuffer(alphas_prev, "sampler/test/test_alphas_prev.npy");
    // util::testBuffer(sqrt_one_minus_alphas, "sampler/test/test_sqrt_one_minus_alphas.npy");

    for (int index = static_cast<int>(ddim_timesteps.size()) - 1; index >= 0; index--) {
        auto step = ddim_timesteps[index];

        img = p_sample_ddim(img, step, conditioning, index, alphas, alphas_prev,
                            sqrt_one_minus_alphas);
        // max diff: 0.00000047683715820312
        // util::testBuffer(img, "sampler/test/test_img_after.npy");
    }

    return img;
}

/**
 * @brief Assume eta=0.0, batch_size=1, unconditional_guidance_scale=1.0, unconditional_conditioning=null
 * @param ddim_num_steps
 */
std::vector<float> DDIMSampler::p_sample_ddim(
        const std::vector<float> &x, int t, const std::vector<float> &c,
        size_t index,
        std::vector<float> &alphas,
        std::vector<float> &alphas_prev,
        std::vector<float> &sqrt_one_minus_alphas) {
    std::vector<float> e_t, dir_xt(x.size()), x_prev(x.size()), pred_x0(x.size());
    float a_t, a_prev, sqrt_one_minus_at;

    a_t = alphas[index];
    a_prev = alphas_prev[index];
    sqrt_one_minus_at = sqrt_one_minus_alphas[index];

    e_t = apply_model(x, t, c);

    for (int i = 0; i < e_t.size(); i++) {
        pred_x0[i] = (x[i] - sqrt_one_minus_at * e_t[i]) / sqrt(a_t);
    }

    for (int i = 0; i < e_t.size(); i++) {
        dir_xt[i] = sqrt(1.0f - a_prev) * e_t[i];
    }

    for (int i = 0; i < e_t.size(); i++) {
        x_prev[i] = sqrt(a_prev) * pred_x0[i] + dir_xt[i];
    }
    return x_prev;
}

std::pair<std::vector<float>, std::vector<float>>
DDIMSampler::make_schedule(const std::vector<int> &ddim_timesteps) {
    std::vector<float> alphas;
    for (auto step: ddim_timesteps) {
        alphas.push_back(alphas_cumprod[step]);
    }

    std::vector<float> alphas_prev;
    alphas_prev.push_back(alphas_cumprod[0]);
    for (int i = 0; i < ddim_timesteps.size() - 1; i++) {
        alphas_prev.push_back(alphas_cumprod[ddim_timesteps[i]]);
    }

    return std::make_pair(alphas, alphas_prev);
}

template<typename T>
std::vector<T> DDIMSampler::linspace(T start, T end, int steps) {
    std::vector<T> result;
    if (steps <= 1) {
        result.push_back(start);
        return result;
    }

    T stepSize = (end - start) / static_cast<T>(steps - 1);

    for (int i = 0; i < steps; ++i) {
        result.push_back(start + static_cast<T>(i) * stepSize);
    }

    return result;
}