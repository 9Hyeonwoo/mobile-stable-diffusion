#include <jni.h>
#include <string>
#include <android/log.h>
#include "modules/tokenizer.h"
#include "modules/TextEncoder.h"
#include "modules/DDIMSampler.h"
#include "modules/UNetModel.h"
#include "modules/Decoder.h"
#include "modules/util.h"
#include <chrono>

#define CL_TARGET_OPENCL_VERSION 200

#include <CL/opencl.h>

#define LOG_TAG "MAIN"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
      __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
      throw std::runtime_error("OpenCL error."); \
    }

cl_context context;
cl_command_queue cmdQueue;
cl_device_id deviceId;

DDIMSampler *sampler;
AAssetManager *assetManager;

extern "C" JNIEXPORT void JNICALL
Java_com_example_myopencl_MainActivity_initOpenCL(
        JNIEnv *env,
        jobject thiz,
        jobject _assetManager) {
    cl_platform_id platformId;
    cl_int err;

    err = clGetPlatformIDs(1, &platformId, nullptr);
    err |= clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
    CHECK_ERROR(err);

    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    CHECK_ERROR(err);

    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                        0};
    cmdQueue = clCreateCommandQueueWithProperties(context, deviceId, properties, &err);
    CHECK_ERROR(err);

    assetManager = AAssetManager_fromJava(env, _assetManager);

    // auto clazz = env->FindClass("com/example/myopencl/MainActivity");
    // auto methodId = env->GetMethodID(clazz, "unet", "([FJ[F)[F");

//    sampler = new DDIMSampler([](std::vector<float> x, int t, std::vector<float> c){
//        // auto _result = (jfloatArray)(env->CallObjectMethod(thiz, methodId, x.data(), t, c.data()));
//        // float* floatArray = env->GetFloatArrayElements(_result, nullptr);
//        return x;
//    });
    // int shape[3] = {4, 64, 64};
    // std::vector<float> tmp_c(77*1024, 0.f);
    // auto x = util::load_npy_file("sampler/test/test_img.npy");
    // auto x_vec = x.as_vec<float>();
    // sampler->sample(&x_vec, 50, shape, tmp_c);

    /* test unet */
//    auto x = util::load_npy_file("sampler/test/test_seed_45_img.npy").as_vec<float>();
//    auto c = util::load_npy_file("encoder/test/ln_final_test_fp32.npy").as_vec<float>();
//    unet->forward(x, 981, c);

    /* test decoder */
//    auto x = util::load_npy_file("decoder/test/test_up_0.npy").as_vec<float>();
//    decoder->test(x);


    size_t maxWorkGroupSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                          &maxWorkGroupSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_GROUP_SIZE: %ld",
                        maxWorkGroupSize);

    cl_uint maxDims;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxDims,
                          nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d",
                        maxDims);

    size_t maxWorkItemSizes[maxDims];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDims,
                          &maxWorkItemSizes, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_WORK_ITEM_SIZES: %ld, %ld, %ld",
                        maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

    char openclVersion[100];
    err = clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, sizeof(openclVersion),
                          &openclVersion, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_OPENCL_C_VERSION: %s",
                        openclVersion);

    cl_ulong localMemSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize,
                          nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_LOCAL_MEM_SIZE(32KB): %ld",
                        localMemSize);

    cl_ulong globalMemSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize,
                          nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_GLOBAL_MEM_SIZE(5.6GB): %ld",
                        globalMemSize);

    cl_device_fp_config fpConfig;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config),
                          &fpConfig, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_SINGLE_FP_CONFIG: %ld", fpConfig);

    cl_device_local_mem_type localMemType;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type),
                          &localMemType, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_LOCAL_MEM_TYPE(CL_GLOBAL): %u", localMemType);

    cl_uint maxComputeUnits;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                          &maxComputeUnits, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_COMPUTE_UNITS(14): %u",
                        maxComputeUnits);

    size_t maxGlobalVariableSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, sizeof(size_t),
                          &maxGlobalVariableSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE(64KB): %ld",
                        maxGlobalVariableSize);

    cl_ulong maxGlobalAllocSize;
    err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                          &maxGlobalAllocSize, nullptr);
    CHECK_ERROR(err);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "CL_DEVICE_MAX_MEM_ALLOC_SIZE(5.6GB): %ld",
                        maxGlobalAllocSize);
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_com_example_myopencl_MainActivity_tokenize(JNIEnv *env, jobject thiz, jstring _text) {
    auto tokenizer = SimpleTokenizer();
    const char *text = env->GetStringUTFChars(_text, nullptr);
    auto result = tokenizer.tokenize(text);
    for (const auto i: result) {
        //__android_log_print(ANDROID_LOG_DEBUG, "__TEST__", "encode: %ld", i);
    }
    env->ReleaseStringUTFChars(_text, text);

    jlongArray resultArray = env->NewLongArray(static_cast<int>(result.size()));
    env->SetLongArrayRegion(resultArray, 0, static_cast<int>(result.size()), result.data());

    return resultArray;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myopencl_MainActivity_encode(JNIEnv *env, jobject thiz, jlongArray _token) {
    auto encoder = TextEncoder(assetManager, context, cmdQueue, deviceId);

    long *longArray = env->GetLongArrayElements(_token, nullptr);
    auto token = std::vector<long>(longArray, longArray + env->GetArrayLength(_token));

    auto start = std::chrono::high_resolution_clock::now();
    auto encodedToken = encoder.encode(token);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "text encoder exec time: %lld ms",
                        duration.count());

    jfloatArray result = env->NewFloatArray(static_cast<int>(encodedToken.size()));
    env->SetFloatArrayRegion(result, 0, static_cast<int>(encodedToken.size()), encodedToken.data());
    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myopencl_MainActivity_sample(JNIEnv *env, jobject thiz, jfloatArray _condition) {
    float *floatArray = env->GetFloatArrayElements(_condition, nullptr);
    auto condition = std::vector<float>(floatArray, floatArray + env->GetArrayLength(_condition));

//    auto x = util::load_npy_file("sampler/test/test_seed_45_img.npy");
//    auto x_vec = x.as_vec<float>();
//    int shape[3] = {4, 64, 64};
//    auto result = sampler->sample(&x_vec, 50, shape, condition);

    auto unet = UNetModel(assetManager, context, cmdQueue, deviceId);
    auto x = util::load_npy_file("sampler/test/test_seed_45_img.npy").as_vec<float>();
    auto c = util::load_npy_file("encoder/test/ln_final_test_fp32.npy").as_vec<float>();
    auto start = std::chrono::high_resolution_clock::now();
    auto result = unet.forward(x, 981, c);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "u-net exec time: %lld ms", duration.count());

//    auto result = util::load_npy_file("sampler/test/test_seed_45_img.npy").as_vec<float>();
//    unet.test(result, 981, c);

    jfloatArray resultArray = env->NewFloatArray(static_cast<int>(result.size()));
    env->SetFloatArrayRegion(resultArray, 0, static_cast<int>(result.size()), result.data());
    return resultArray;

}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myopencl_MainActivity_decode(JNIEnv *env, jobject thiz) {
    auto decoder = Decoder(context, cmdQueue, deviceId, assetManager);

    auto x = util::load_npy_file("decoder/test/test_seed_45_step_50_sample.npy").as_vec<float>();
    auto start = std::chrono::high_resolution_clock::now();
    auto result = decoder.decode(x);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "decoder exec time: %lld ms", duration.count());

//    auto result = util::load_npy_file("decoder/test/test_mid_block_1.npy").as_vec<float>();
//    decoder.test(result);

    jfloatArray resultArray = env->NewFloatArray(static_cast<jint>(result.size()));
    env->SetFloatArrayRegion(resultArray, 0, static_cast<jint>(result.size()), result.data());
    return resultArray;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myopencl_MainActivity_destroyOpenCL(JNIEnv *env, jobject thiz) {
    delete sampler;

    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
}