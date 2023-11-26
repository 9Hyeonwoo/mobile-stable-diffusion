//
// Created by 구현우 on 2023/11/25.
//

#ifndef MY_OPENCL_UTIL_H
#define MY_OPENCL_UTIL_H

#include <vector>

namespace util {
    template<typename T>
    std::vector <T> permute(const std::vector <T> &vec, const std::vector <size_t> &shape, const std::vector <int> &dimensions);

}
#endif //MY_OPENCL_UTIL_H
