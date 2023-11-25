//
// Created by 구현우 on 2023/11/24.
//

#include "TextEncoder.h"

#define LOG_TAG "TEXT_ENCODER"
#define EMBEDDING_SIZE 1024

cnpy::NpyArray* load_npy_file(const unsigned char* buffer, off_t uncompr_bytes) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(buffer,word_size,shape,fortran_order);

    auto *arr = new cnpy::NpyArray(shape, word_size, fortran_order);
    size_t offset = uncompr_bytes - arr->num_bytes();
    memcpy(arr->data<char>(), buffer+offset, arr->num_bytes());
    return arr;
}

TextEncoder::TextEncoder(AAssetManager *assetManager) {
    AAsset *asset = AAssetManager_open(assetManager, "encoder/embedding_fp32.npy", AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        return;
    }

    auto buffer = static_cast<const unsigned char*>(AAsset_getBuffer(asset));
    auto length = AAsset_getLength(asset);
    embedding = load_npy_file(buffer, length);

    AAsset_close(asset);
}

TextEncoder::~TextEncoder() {
    delete embedding;
}

/*
 * @input: `token` tokenized text
 * @return: token embedding with size=(length of 'token' * EMBEDDING_SIZE)
 */
std::vector<float> TextEncoder::token_embedding(std::vector<long> token) {
    std::vector<float> result;
    for (auto i: token) {
        auto data = embedding->data<float>() + (i * EMBEDDING_SIZE);
        result.insert(result.end(), data, data + EMBEDDING_SIZE);
    }
    return result;
}