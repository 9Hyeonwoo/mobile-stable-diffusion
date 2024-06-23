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

/**
 * A                       B
 * [-][-][-][>][ ]...[ ]   [|][ ][ ][ ][ ]...[ ]
 * [ ][ ][ ][ ][ ]...[ ]   [|][ ][ ][ ][ ]...[ ]
 * [ ][ ][ ][ ][ ]...[ ] X [|][ ][ ][ ][ ]...[ ]
 * [ ][ ][ ][ ][ ]...[ ]   [V][ ][ ][ ][ ]...[ ]
 *  :  :  :  :  :  :  :     :  :  :  :  :  :  :
 * [ ][ ][ ][ ][ ]...[ ]   [ ][ ][ ][ ][ ]...[ ]
 **/
__kernel void optimized_einsum_bik_bkj_bij(
    __global floatX *A,
    __global floatX *B,
    __global float *C,
    const int K
) {

    const int reg_size_m = 4;
    const int local_size_i = get_local_size(1);
    const int local_size_j = get_local_size(2);

    const int group_i = get_group_id(1);
    const int group_j = get_group_id(2);

    const int local_id_i = get_local_id(1);
    const int local_id_j = get_local_id(2);

    const int M = get_global_size(1) * reg_size_m;
    const int N_div_width = get_global_size(2);
    const int bi_offset = get_global_id(0) * M;
    const int bj_offset = get_global_id(0) * K * N_div_width;

    int i = bi_offset + group_i * local_size_i * reg_size_m + local_id_i;

    float acc[reg_size_m][WIDTH];

    for (int wm=0; wm<reg_size_m; wm++) {
        for (int wn=0; wn<WIDTH; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    floatX vecA[reg_size_m], vecB;
    const int K_div_width = K / WIDTH;
    for (int k = 0; k < K; k++) {
        if (k % WIDTH == 0) {
            for (int wm=0; wm<reg_size_m; wm++) {
                vecA[wm] = A[(i + wm * local_size_i) * K_div_width + (k / WIDTH)];
            }
        }
        vecB = B[bj_offset + k * N_div_width + group_j * local_size_j + local_id_j];
        for (int wm=0; wm<reg_size_m; wm++) {
            float A;
#if WIDTH == 1
            acc[wm][0] += vecA[wm] * vecB;
#elif WIDTH == 2
            if (k % WIDTH == 0) {
                A = vecA[wm].x;
            } else {
                A = vecA[wm].y;
            }
            acc[wm][0] += A * vecB.x;
            acc[wm][1] += A * vecB.y;
#elif WIDTH == 4
            if (k % WIDTH == 0) {
                A = vecA[wm].x;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].y;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].z;
            } else {
                A = vecA[wm].w;
            }
            acc[wm][0] += A * vecB.x;
            acc[wm][1] += A * vecB.y;
            acc[wm][2] += A * vecB.z;
            acc[wm][3] += A * vecB.w;
#elif WIDTH == 8
            if (k % WIDTH == 0) {
                A = vecA[wm].s0;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].s1;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].s2;
            } else if (k % WIDTH == 3) {
                A = vecA[wm].s3;
            } else if (k % WIDTH == 4) {
                A = vecA[wm].s4;
            } else if (k % WIDTH == 5) {
                A = vecA[wm].s5;
            } else if (k % WIDTH == 6) {
                A = vecA[wm].s6;
            } else {
                A = vecA[wm].s7;
            }
            acc[wm][0] += A * vecB.s0;
            acc[wm][1] += A * vecB.s1;
            acc[wm][2] += A * vecB.s2;
            acc[wm][3] += A * vecB.s3;
            acc[wm][4] += A * vecB.s4;
            acc[wm][5] += A * vecB.s5;
            acc[wm][6] += A * vecB.s6;
            acc[wm][7] += A * vecB.s7;
#elif WIDTH == 16
            if (k % WIDTH == 0) {
                A = vecA[wm].s0;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].s1;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].s2;
            } else if (k % WIDTH == 3) {
                A = vecA[wm].s3;
            } else if (k % WIDTH == 4) {
                A = vecA[wm].s4;
            } else if (k % WIDTH == 5) {
                A = vecA[wm].s5;
            } else if (k % WIDTH == 6) {
                A = vecA[wm].s6;
            } else if (k % WIDTH == 7) {
                A = vecA[wm].s7;
            } else if (k % WIDTH == 8) {
                A = vecA[wm].s8;
            } else if (k % WIDTH == 9) {
                A = vecA[wm].s9;
            } else if (k % WIDTH == 10) {
                A = vecA[wm].sA;
            } else if (k % WIDTH == 11) {
                A = vecA[wm].sB;
            } else if (k % WIDTH == 12) {
                A = vecA[wm].sC;
            } else if (k % WIDTH == 13) {
                A = vecA[wm].sD;
            } else if (k % WIDTH == 14) {
                A = vecA[wm].sE;
            } else {
                A = vecA[wm].sF;
            }
            acc[wm][0] += A * vecB.s0;
            acc[wm][1] += A * vecB.s1;
            acc[wm][2] += A * vecB.s2;
            acc[wm][3] += A * vecB.s3;
            acc[wm][4] += A * vecB.s4;
            acc[wm][5] += A * vecB.s5;
            acc[wm][6] += A * vecB.s6;
            acc[wm][7] += A * vecB.s7;
            acc[wm][8] += A * vecB.s8;
            acc[wm][9] += A * vecB.s9;
            acc[wm][10] += A * vecB.sA;
            acc[wm][11] += A * vecB.sB;
            acc[wm][12] += A * vecB.sC;
            acc[wm][13] += A * vecB.sD;
            acc[wm][14] += A * vecB.sE;
            acc[wm][15] += A * vecB.sF;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm=0; wm<reg_size_m; wm++) {
        for (int wn=0; wn<WIDTH; wn++) {
            C[(i + wm * local_size_i) * (N_div_width * WIDTH) + (group_j * local_size_j * WIDTH + local_id_j*WIDTH+wn)] = acc[wm][wn];
        }
    }
}

__kernel void optimized_einsum_bik_bkj_bij_general(
    __global floatX *A,
    __global floatX *B,
    __global float *C,
    const int N,
    const int K,
    const float scale
) {

    const int reg_size_m = 4;
    const int local_size_i = get_local_size(1);
    const int local_size_j = get_local_size(2);

    const int group_i = get_group_id(1);
    const int group_j = get_group_id(2);

    const int local_id_i = get_local_id(1);
    const int local_id_j = get_local_id(2);

    const int M = get_global_size(1) * reg_size_m;
    const int N_div_width = get_global_size(2);

    int b = get_global_id(0);
    int i = group_i * local_size_i * reg_size_m + local_id_i;
    int j_div_WIDTH = group_j * local_size_j + local_id_j;

    float acc[reg_size_m][WIDTH];

    for (int wm=0; wm<reg_size_m; wm++) {
        for (int wn=0; wn<WIDTH; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    floatX vecA[reg_size_m], vecB;
    const int K_div_width = K / WIDTH;
    for (int k = 0; k < K; k++) {
        if (k % WIDTH == 0) {
            for (int wm=0; wm<reg_size_m; wm++) {
                vecA[wm] = A[((b * M) + i + wm * local_size_i) * K_div_width + (k / WIDTH)];
            }
        }
        vecB = B[((b * K) + k) * N_div_width + j_div_WIDTH];
        for (int wm=0; wm<reg_size_m; wm++) {
            float A;
#if WIDTH == 1
            acc[wm][0] += vecA[wm] * vecB;
#elif WIDTH == 2
            if (k % WIDTH == 0) {
                A = vecA[wm].x;
            } else {
                A = vecA[wm].y;
            }
            acc[wm][0] += A * vecB.x;
            acc[wm][1] += A * vecB.y;
#elif WIDTH == 4
            if (k % WIDTH == 0) {
                A = vecA[wm].x;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].y;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].z;
            } else {
                A = vecA[wm].w;
            }
            acc[wm][0] += A * vecB.x;
            acc[wm][1] += A * vecB.y;
            acc[wm][2] += A * vecB.z;
            acc[wm][3] += A * vecB.w;
#elif WIDTH == 8
            if (k % WIDTH == 0) {
                A = vecA[wm].s0;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].s1;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].s2;
            } else if (k % WIDTH == 3) {
                A = vecA[wm].s3;
            } else if (k % WIDTH == 4) {
                A = vecA[wm].s4;
            } else if (k % WIDTH == 5) {
                A = vecA[wm].s5;
            } else if (k % WIDTH == 6) {
                A = vecA[wm].s6;
            } else {
                A = vecA[wm].s7;
            }
            acc[wm][0] += A * vecB.s0;
            acc[wm][1] += A * vecB.s1;
            acc[wm][2] += A * vecB.s2;
            acc[wm][3] += A * vecB.s3;
            acc[wm][4] += A * vecB.s4;
            acc[wm][5] += A * vecB.s5;
            acc[wm][6] += A * vecB.s6;
            acc[wm][7] += A * vecB.s7;
#elif WIDTH == 16
            if (k % WIDTH == 0) {
                A = vecA[wm].s0;
            } else if (k % WIDTH == 1) {
                A = vecA[wm].s1;
            } else if (k % WIDTH == 2) {
                A = vecA[wm].s2;
            } else if (k % WIDTH == 3) {
                A = vecA[wm].s3;
            } else if (k % WIDTH == 4) {
                A = vecA[wm].s4;
            } else if (k % WIDTH == 5) {
                A = vecA[wm].s5;
            } else if (k % WIDTH == 6) {
                A = vecA[wm].s6;
            } else if (k % WIDTH == 7) {
                A = vecA[wm].s7;
            } else if (k % WIDTH == 8) {
                A = vecA[wm].s8;
            } else if (k % WIDTH == 9) {
                A = vecA[wm].s9;
            } else if (k % WIDTH == 10) {
                A = vecA[wm].sA;
            } else if (k % WIDTH == 11) {
                A = vecA[wm].sB;
            } else if (k % WIDTH == 12) {
                A = vecA[wm].sC;
            } else if (k % WIDTH == 13) {
                A = vecA[wm].sD;
            } else if (k % WIDTH == 14) {
                A = vecA[wm].sE;
            } else {
                A = vecA[wm].sF;
            }
            acc[wm][0] += A * vecB.s0;
            acc[wm][1] += A * vecB.s1;
            acc[wm][2] += A * vecB.s2;
            acc[wm][3] += A * vecB.s3;
            acc[wm][4] += A * vecB.s4;
            acc[wm][5] += A * vecB.s5;
            acc[wm][6] += A * vecB.s6;
            acc[wm][7] += A * vecB.s7;
            acc[wm][8] += A * vecB.s8;
            acc[wm][9] += A * vecB.s9;
            acc[wm][10] += A * vecB.sA;
            acc[wm][11] += A * vecB.sB;
            acc[wm][12] += A * vecB.sC;
            acc[wm][13] += A * vecB.sD;
            acc[wm][14] += A * vecB.sE;
            acc[wm][15] += A * vecB.sF;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm=0; wm<reg_size_m; wm++) {
        for (int wn=0; wn<WIDTH; wn++) {
            int j = j_div_WIDTH * WIDTH + wn;
            if (j < N) {
                C[((b * M) + (i + wm * local_size_i)) * N + j] = acc[wm][wn] * scale;
            }
        }
    }
}