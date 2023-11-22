//
// Created by 구현우 on 2023/11/19.
//

#ifndef MY_OPENCL_TOKENIZER_H
#define MY_OPENCL_TOKENIZER_H

#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <string>
#include <vector>
#include <regex>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <map>
#include <limits>
#include <numeric>
#include <locale>
#include <codecvt>

#define LOG_TAG "TOKENIZER"
#define CONTEXT_LENGTH 77

class SimpleTokenizer {
public:
    SimpleTokenizer(AAssetManager *assetManager);
    ~SimpleTokenizer();

    std::vector<long> tokenize(const std::vector<std::string>& texts);
    std::vector<long> tokenize(const std::string& text);
private:
    std::regex pat;
    std::unordered_map<int, std::string> byte_encoder;
    std::unordered_map<std::string, int> encoder;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::unordered_map<std::string, std::string> cache;

    static std::vector<std::pair<std::string, std::string>> get_pairs(std::vector<std::string> word);
    static std::vector<std::pair<int, std::string>> bytes_to_unicode();
    std::string bpe(std::string token);
    std::vector<int> encode(std::string text);
};

#endif //MY_OPENCL_TOKENIZER_H
