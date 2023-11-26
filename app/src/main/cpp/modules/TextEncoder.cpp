//
// Created by 구현우 on 2023/11/24.
//

#include "TextEncoder.h"

#include "util.h"

#define LOG_TAG "TEXT_ENCODER"
#define EMBEDDING_SIZE 1024

cnpy::NpyArray* load_npy_file(AAssetManager* assetManager, const char *filename) {
    AAsset *asset = AAssetManager_open(assetManager, filename, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        throw std::runtime_error("Failed to open the asset.");
    }

    auto buffer = static_cast<const unsigned char*>(AAsset_getBuffer(asset));
    auto length = AAsset_getLength(asset);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(buffer,word_size,shape,fortran_order);

    auto arr = new cnpy::NpyArray(shape, word_size, fortran_order);
    size_t offset = length - arr->num_bytes();
    memcpy(arr->data<char>(), buffer+offset, arr->num_bytes());

    AAsset_close(asset);
    return arr;
}

TextEncoder::TextEncoder(AAssetManager *assetManager) {
    embedding = load_npy_file(assetManager, "encoder/embedding_fp32.npy");
    positional_embedding = load_npy_file(assetManager, "encoder/positional_embedding_fp32.npy");

}

TextEncoder::~TextEncoder() {
    delete embedding;
    delete positional_embedding;
}

/*
 * @input: `token` tokenized text
 * @return: token embedding with size=(length of 'token'(CONTEXT_LENGTH=77) * (EMBEDDING_SIZE=1024))
 */
std::vector<float> TextEncoder::token_embedding(const std::vector<long>& token) {
    std::vector<float> result;
    for (auto i: token) {
        auto data = embedding->data<float>() + (i * EMBEDDING_SIZE);
        result.insert(result.end(), data, data + EMBEDDING_SIZE);
    }
    return result;
}

std::vector<float> TextEncoder::encode(const std::vector<long> &token) {
    std::vector<float> result;
    auto token_embedding_result = token_embedding(token);

    for (int i=0; i < token_embedding_result.size(); i++) {
        token_embedding_result[i] += positional_embedding->data<float>()[i];
    }

    auto permute_result = util::permute<float>(positional_embedding->as_vec<float>(), positional_embedding->shape, {1, 0, 2});

    // TODO : text_transformer_forward(x)

    // TODO : x.permute(1, 0, 2)

    // TODO : ln_final(x)

    return result;
}