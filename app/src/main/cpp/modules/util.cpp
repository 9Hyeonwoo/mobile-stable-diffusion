//
// Created by 구현우 on 2023/11/25.
//

#include "util.h"

#include <algorithm>
#include <android/log.h>
#include <numeric>

#define LOG_TAG "UTIL"

template<typename T>
std::vector <T> util::permute(const std::vector <T> &vec, const std::vector <size_t> &shape, const std::vector <int> &dimensions) {
    if (shape.size() != dimensions.size() || dimensions.size() != 3) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Shape and dimensions size mismatch.");
        throw std::runtime_error("Shape and dimensions size mismatch.");
    }

    for (int i=0; i < shape.size(); i++) {
        if (std::find(dimensions.begin(), dimensions.end(), i) == dimensions.end()) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Cannot find %d dimension in dimensions", i);
            throw std::runtime_error("Cannot find dimension in dimensions");
        }
    }

    size_t sum = 1;
    for (auto i: shape) {
        sum *= i;
    }
    if (sum != vec.size()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Shape and vector size mismatch.");
        throw std::runtime_error("Shape and vector size mismatch.");
    }

    if (dimensions.size() == 3) {
        std::vector<T> result(vec.size());
        for (int i=0; i < shape[0]; i++) {
            for (int j=0; j < shape[1]; j++) {
                for (int k=0; k < shape[2]; k++) {
                    size_t index = 0;
                    size_t multiplier = shape[dimensions[1]] * shape[dimensions[2]];
                    for (int d=0; d < dimensions.size(); ++d) {
                        switch (dimensions[d]) {
                            case 0:
                                index += i * multiplier;
                                break;
                            case 1:
                                index += j * multiplier;
                                break;
                            case 2:
                            default:
                                index += k * multiplier;
                                break;
                        }
                        multiplier /= shape[dimensions[d+1]];
                    }

                    // result[dimensions[0]][dimensions[1]][dimensions[2]] = vec[i][j][k]
                    result[index] = vec[i * shape[1] * shape[2] + j * shape[2] + k];
                }
            }
        }
        return result;
    }

    return vec;
}