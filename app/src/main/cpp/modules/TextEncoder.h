//
// Created by 구현우 on 2023/11/24.
//

#ifndef MY_OPENCL_TEXTENCODER_H
#define MY_OPENCL_TEXTENCODER_H

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "cnpy.h"
#include <vector>
#include <stdio.h>

class TextEncoder {
    public:
    TextEncoder(AAssetManager *assetManager);
    ~TextEncoder();
private:
    std::vector<float> token_embedding(const std::vector<long>& token);

    cnpy::NpyArray* embedding;
    cnpy::NpyArray* positional_embedding;
};


#endif //MY_OPENCL_TEXTENCODER_H
