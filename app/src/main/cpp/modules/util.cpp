//
// Created by 구현우 on 2023/11/25.
//

#include "util.h"

#include <algorithm>
#include <android/log.h>
#include <numeric>
#include <cstdio>
#include <fstream>

#define LOG_TAG "UTIL"

#define MEDIA_PATH "/sdcard/Android/media/com.example.myopencl/"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

std::vector<float> *
util::permute3D(const std::vector<float> &vec, const int shape[3], const int dimensions[3]) {
    auto result = new std::vector<float>(vec.size(), 0);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                size_t index = 0;
                size_t multiplier = shape[dimensions[1]] * shape[dimensions[2]];
                for (int d = 0; d < 3; ++d) {
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
                    if (d < 2) {
                        multiplier /= shape[dimensions[d + 1]];
                    }
                }

                // result[dimensions[0]][dimensions[1]][dimensions[2]] = vec[i][j][k]
                auto vecIndex = i * shape[1] * shape[2] + j * shape[2] + k;
                (*result)[index] = vec[vecIndex];
            }
        }
    }
    return result;
}

cl_program util::create_and_build_program_with_source(cl_context context,
                                                      cl_device_id device,
                                                      AAssetManager *assetManager,
                                                      const char *file_name) {
    AAsset *asset = AAssetManager_open(assetManager, file_name, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        throw std::runtime_error("Failed to open the asset.");
    }

    auto buffer = static_cast<const unsigned char *>(AAsset_getBuffer(asset));
    size_t source_size = AAsset_getLength(asset);

    char *source_code = (char *) malloc(source_size + 1);
    memcpy(source_code, buffer, source_size);
    source_code[source_size] = '\0';
    AAsset_close(asset);

    cl_int err;
    cl_program program = clCreateProgramWithSource(
            context, 1, (const char **) &source_code, &source_size, &err);
    CHECK_ERROR(err);
    free(source_code);
    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                          nullptr, &log_size));
        char *log = (char *) malloc(log_size + 1);
        CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                          log_size, log, nullptr));
        log[log_size] = 0;
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Compile error:\n%s", log);
        free(log);
    }
    CHECK_ERROR(err);
    return program;
}

cnpy::NpyArray util::load_npy_file(const std::string &_filename) {
    auto filename = MEDIA_PATH + _filename;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the file. %s",
                            filename.data());
        throw std::runtime_error("Failed to open the file.");
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Create a vector to hold the file content
    std::vector<unsigned char> _buffer(fileSize);

    // Read the file into the buffer
    file.read(reinterpret_cast<char *>(_buffer.data()), fileSize);

    // Close the file
    file.close();

    auto buffer = static_cast<const unsigned char *>(_buffer.data());

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(buffer, word_size, shape, fortran_order);

    auto arr = cnpy::NpyArray(shape, word_size, fortran_order);
    size_t offset = _buffer.size() - arr.num_bytes();
    memcpy(arr.data<char>(), buffer + offset, arr.num_bytes());

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "load_npy_file: %s", _filename.c_str());
    return arr;
}

void util::testBuffer(
        cl_command_queue cmdQueue, cl_mem buffer, const char *filename
) {
    cl_int err;

    size_t bufferBytes;
    err = clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size_t), &bufferBytes, nullptr);
    CHECK_ERROR(err);
    auto bufferSize = bufferBytes / sizeof(float);

    std::vector<float> result(bufferSize);
    err = clEnqueueReadBuffer(cmdQueue, buffer, CL_TRUE, 0,
                              sizeof(float) * bufferSize,
                              result.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err);

    util::testBuffer(result, filename);
}

void util::testBuffer(std::vector<float> result, const char *filename) {
    auto test = util::load_npy_file(filename);
    if (result.size() != test.num_vals) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s: bufferSize(%ld) != test.num_vals(%ld)",
                            filename, result.size(), test.num_vals);
        return;
    }

    float maxDiff = 0;
    int maxId = 0;
    std::vector<int> wrongs;
    for (int i = 0; i < test.num_vals; i++) {
        if (result[i] != test.data<float>()[i]) {
            auto diff = std::abs(result[i] - test.data<float>()[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
                maxId = i;
                wrongs.insert(wrongs.begin(), i);
            } else {
                wrongs.push_back(i);
            }
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "%s max diff: %.20f / num : %ld / result[%d]: %f/ test[%d]: %f",
                        filename, maxDiff, wrongs.size(), maxId, result[maxId], maxId,
                        test.data<float>()[maxId]);
    for (int i = 0; i < 10; i++) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                            "result[%d]: %f != test[%d]: %f",
                            wrongs[i], result[wrongs[i]], wrongs[i], test.data<float>()[wrongs[i]]);
    }
}