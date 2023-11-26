//
// Created by 구현우 on 2023/11/25.
//

#ifndef MY_OPENCL_UTIL_H
#define MY_OPENCL_UTIL_H

#include <vector>
#include <android/asset_manager_jni.h>

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>

namespace util {
    template<typename T>
    std::vector<T> permute(const std::vector<T> &vec, const std::vector<size_t> &shape,
                           const std::vector<int> &dimensions);

    cl_program create_and_build_program_with_source(cl_context context, cl_device_id device,
                                                    AAssetManager *assetManager,
                                                    const char *file_name);
}
#endif //MY_OPENCL_UTIL_H
