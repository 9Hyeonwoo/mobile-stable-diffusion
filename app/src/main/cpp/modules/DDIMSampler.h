//
// Created by 구현우 on 2023/12/05.
//

#ifndef MY_OPENCL_DDIMSAMPLER_H
#define MY_OPENCL_DDIMSAMPLER_H

#include <vector>

class DDIMSampler {
public:
    DDIMSampler(const std::function<std::vector<float>(const std::vector<float> &, int,
                                                       const std::vector<float> &)> &apply_model);

    ~DDIMSampler();

    std::vector<float> sample(
            std::vector<float> *x_T,
            int ddim_num_steps,
            const int shape[3],
            const std::vector<float> &conditioning);

private:
    std::vector<float>
    p_sample_ddim(const std::vector<float> &x, int t, const std::vector<float> &c, size_t index,
                  std::vector<float> &alphas,
                  std::vector<float> &alphas_prev,
                  std::vector<float> &sqrt_one_minus_alphas);

    std::pair<std::vector<float>, std::vector<float>>
    make_schedule(const std::vector<int> &ddim_timesteps);

    template<typename T>
    static std::vector<T> linspace(T start, T end, int steps);

    std::vector<float> alphas_cumprod;

    const std::function<std::vector<float>(const std::vector<float> &, int,
                                           const std::vector<float> &)> apply_model;
};


#endif //MY_OPENCL_DDIMSAMPLER_H
