//
// Created by 구현우 on 2023/12/07.
//

#ifndef MY_OPENCL_UNETMODEL_H
#define MY_OPENCL_UNETMODEL_H

#include <vector>

class UNetModel {
public:
    UNetModel() = default;

    ~UNetModel() = default;

    std::vector<float>
    forward(const std::vector<float> &x, long timestep, const std::vector<float> &condition);

private:
    std::vector<float> timestep_embedding(long timestep);
};


#endif //MY_OPENCL_UNETMODEL_H
