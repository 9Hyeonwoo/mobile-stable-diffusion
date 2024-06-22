// b 차원이 추가된 (i, k) * (k, j) -> (i, j)
__kernel void einsum_bik_bjk_bij(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const size_t kSize,
    const float scale
) {
    int b = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    int iSize = get_global_size(1);
    int jSize = get_global_size(2);

    float sum = 0.0f;
    for (int k = 0; k < kSize; k++) {
        sum += A[(b * iSize * kSize) + (i * kSize) + k] * B[(b * jSize * kSize) + (j * kSize) + k];
    }

    C[(b * iSize * jSize) + (i * jSize) + j] = sum * scale;
}

// (i, j) * (j, k) -> (i, k) 형태의 행렬곱.
__kernel void einsum_bij_bjk_bik(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const size_t jSize
) {
    int b = get_global_id(0);
    int i = get_global_id(1);
    int k = get_global_id(2);

    int iSize = get_global_size(1);
    int kSize = get_global_size(2);

    float sum = 0.0f;
    for (int j = 0; j < jSize; j++) {
        sum += A[(b * iSize * jSize) + (i * jSize) + j] * B[(b * jSize * kSize) + (j * kSize) + k];
    }

    C[(b * iSize * kSize) + (i * kSize) + k] = sum;
}

// tiled local, register, vector
#define WIDTH 4
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#elif WIDTH == 16
    typedef float16 floatX;
#endif

__kernel void optimized_einsum_bik_bjk_bij(
    __global floatX *A,
    __global floatX *B,
    __global float *C,
    const int K,
    const float scale
) {

    const int reg_size_m = 8;
    const int local_size_i = get_local_size(1);
    const int local_size_j = get_local_size(2);

    const int group_i = get_group_id(1);

    const int local_id_i = get_local_id(1);

    const int M = get_global_size(1) * reg_size_m;
    const int N = get_global_size(2);
    const int bi_offset = get_global_id(0) * M;
    const int bj_offset = get_global_id(0) * N;

    int i = bi_offset + group_i * local_size_i * reg_size_m + local_id_i ;
    int j = bj_offset + get_global_id(2);

    float acc[reg_size_m];

    for (int wm=0; wm<reg_size_m; wm++) {
        acc[wm] = 0.0f;
    }

    floatX vecA, vecB;
    const int K_div_width = K / WIDTH;
    for (int k = 0; k < K_div_width; k++) {
        vecB = B[j * K_div_width + k];
        for (int wn=0; wn<reg_size_m; wn++) {
            vecA = A[(i + wn * local_size_i) * K_div_width + k];
#if WIDTH == 1
            acc[wn] += vecA * vecB;
#elif WIDTH == 2
            acc[wn] += vecA.x * vecB.x;
            acc[wn] += vecA.y * vecB.y;
#elif WIDTH == 4
            acc[wn] += vecA.x * vecB.x;
            acc[wn] += vecA.y * vecB.y;
            acc[wn] += vecA.z * vecB.z;
            acc[wn] += vecA.w * vecB.w;
#elif WIDTH == 8
            acc[wn] += vecA.s0 * vecB.s0;
            acc[wn] += vecA.s1 * vecB.s1;
            acc[wn] += vecA.s2 * vecB.s2;
            acc[wn] += vecA.s3 * vecB.s3;
            acc[wn] += vecA.s4 * vecB.s4;
            acc[wn] += vecA.s5 * vecB.s5;
            acc[wn] += vecA.s6 * vecB.s6;
            acc[wn] += vecA.s7 * vecB.s7;
#elif WIDTH == 16
            acc[wn] += vecA.s0 * vecB.s0;
            acc[wn] += vecA.s1 * vecB.s1;
            acc[wn] += vecA.s2 * vecB.s2;
            acc[wn] += vecA.s3 * vecB.s3;
            acc[wn] += vecA.s4 * vecB.s4;
            acc[wn] += vecA.s5 * vecB.s5;
            acc[wn] += vecA.s6 * vecB.s6;
            acc[wn] += vecA.s7 * vecB.s7;
            acc[wn] += vecA.s8 * vecB.s8;
            acc[wn] += vecA.s9 * vecB.s9;
            acc[wn] += vecA.sA * vecB.sA;
            acc[wn] += vecA.sB * vecB.sB;
            acc[wn] += vecA.sC * vecB.sC;
            acc[wn] += vecA.sD * vecB.sD;
            acc[wn] += vecA.sE * vecB.sE;
            acc[wn] += vecA.sF * vecB.sF;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wn=0; wn<reg_size_m; wn++) {
        C[(i + wn * local_size_i) * N + get_global_id(2)] = acc[wn] * scale;
    }
}