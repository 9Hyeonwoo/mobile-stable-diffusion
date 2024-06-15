__kernel void linear(__global float *A,
                    __global float *B,
                    __global float *bias,
                    __global float *C,
                    const int K)
{
    int M = get_global_size(0);
    int N = get_global_size(1);

    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A(M, K) * B(N, K) = C(M, N)
        sum += A[i * K + k] * B[j * K + k];
    }
    if (bias != NULL) {
        sum += bias[j];
    }
    C[i * N + j] = sum;
}

// Use 2D register blocking (further increase in work per thread)
// condition :
// K % TSK == 0, M % TSM == 0, N % TSN == 0
__kernel void reg_linear(
    __global const float* A, // A = (M, K)
    __global const float* B, // B = (N, K)
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const uchar reg_size_m,
    const uchar tile_size_m,
    const uchar tile_size_n,
    __local float* Asub,
    __local float* Bsub
    // const size_t tile_size_k
) {

    __constant int reg_size_n = 8;
    __constant int tile_size_k = 16;

    // Thread identifiers
    int local_size_m = get_local_size(0);
    int local_size_n = get_local_size(1);
    const int local_m = get_local_id(0);
    const int local_n = get_local_id(1);
    const int offset_m = get_group_id(0) * get_local_size(0) * reg_size_m ;
    const int offset_n = get_group_id(1) * get_local_size(1) * reg_size_n ;

    // Local memory to fit a tile of A and B
    // __local float Asub[tile_size_m][tile_size_k];
    // __local float Bsub[tile_size_k * tile_size_n];

    // Allocate register space
    float Areg;
    float Breg[reg_size_n];
    float acc[8][reg_size_n];

    // Initialise the accumulation registers
    int wm_size = 0;
    int wn_size = 0;
    for (int wm=0; wm<reg_size_m; wm++) {
        if (offset_m + local_m + wm*local_size_m < M) {
            wm_size = wm + 1;
        } else {
            break;
        }
        for (int wn=0; wn<reg_size_n; wn++) {
            if (offset_n + local_n + wn*local_size_n < N) {
                wn_size = wn + 1;
            } else {
                break;
            }
            acc[wm][wn] = 0.0f;
        }
    }

    if (wm_size == 0 || wn_size == 0) {
        return;
    }

    // Loop over all tiles
    int numTiles = K/tile_size_k;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int la=0; la<(tile_size_m * tile_size_k / local_size_m / local_size_n); la++) {
            int tid = local_m*local_size_n + local_n;
            int id = la*local_size_m*local_size_n + tid;
            int m = id / tile_size_k;
            int k = id % tile_size_k;
            int tiledIndex = tile_size_k*t + k;
            // A = (M, K), B = (K, N)
            int A_index = (offset_m + m)*K + (tiledIndex);
            if (A_index < M*K) {
                Asub[m*tile_size_k + k] = A[A_index];
            } else {
                break;
            }
        }
        for (int id=local_m*local_size_n + local_n; id<(tile_size_n * tile_size_k); id+=local_size_m*local_size_n) {
            int n = id / tile_size_k;
            int k = id % tile_size_k;
            int tiledIndex = tile_size_k*t + k;
            // A = (M, K), B = (K, N)
            int B_index = (offset_n + n)*K + (tiledIndex);
            if (B_index < N*K) {
                Bsub[k*tile_size_n + n] = B[B_index];
            } else {
                break;
            }
        }
        // global version
        //__global const float* Asub_global = A + offset_m * K + t * tile_size_k;
        //__global const float* Bsub_global = B + offset_n * K + t * tile_size_k;

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<tile_size_k; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn<wn_size; wn++) {
                int col = local_n + wn*local_size_n;
                Breg[wn] = Bsub[k*tile_size_n + col];
                //Breg[wn] = Bsub_global[col * K + k];
            }

            // Perform the computation

            for (int wm=0; wm<wm_size; wm++) {
                int row = local_m + wm*local_size_m;
                Areg = Asub[row*tile_size_k + k];
                //Areg = Asub_global[row*K + k];
                for (int wn=0; wn<wn_size; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wm=0; wm<wm_size; wm++) {
        int global_m = offset_m + local_m + wm*local_size_m;
        for (int wn=0; wn<wn_size; wn++) {
            int global_n = offset_n + local_n + wn*local_size_n;
            if (bias != NULL) {
                acc[wm][wn] += bias[global_n];
            }
            C[global_m*N + global_n] = acc[wm][wn];
        }
    }
}

// remove memory copy in the reg_linear
__kernel void reg_linear_v2(
    __global const float* A, // A = (M, K)
    __global const float* B, // B = (N, K)
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const uchar reg_size_m,
    const uchar tile_size_m,
    const uchar tile_size_n,
    __local float* Asub,
    __local float* Bsub
    // const size_t tile_size_k
) {

    __constant int reg_size_n = 8;
    __constant int tile_size_k = 16;

    // Thread identifiers
    int local_size_m = get_local_size(0);
    int local_size_n = get_local_size(1);
    const int local_m = get_local_id(0);
    const int local_n = get_local_id(1);
    const int offset_m = get_group_id(0) * get_local_size(0) * reg_size_m ;
    const int offset_n = get_group_id(1) * get_local_size(1) * reg_size_n ;

    // Local memory to fit a tile of A and B
    // __local float Asub[tile_size_m][tile_size_k];
    // __local float Bsub[tile_size_k * tile_size_n];

    // Allocate register space
    float Areg;
    float Breg[reg_size_n];
    float acc[8][reg_size_n];

    // Initialise the accumulation registers
    int wm_size = reg_size_m;
    int wn_size = reg_size_n;
    for (int wm=0; wm<reg_size_m; wm++) {
        for (int wn=0; wn<reg_size_n; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    int numTiles = K/tile_size_k;
    for (int t=0; t<numTiles; t++) {

        int a = offset_m * K + t * tile_size_k;
        int b = offset_n * K + t * tile_size_k;

        // Loop over the values of a single tile
        for (int k=0; k<tile_size_k; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn<wn_size; wn++) {
                int col = local_n + wn*local_size_n;
                // Breg[wn] = Bsub[k*tile_size_n + col];
                Breg[wn] = B[b + (col*K + k)];
                //Breg[wn] = Bsub_global[col * K + k];
            }

            // Perform the computation

            for (int wm=0; wm<wm_size; wm++) {
                int row = local_m + wm*local_size_m;
                // Areg = Asub[row*tile_size_k + k];
                Areg = A[a + (row*K + k)];
                //Areg = Asub_global[row*K + k];
                for (int wn=0; wn<wn_size; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wm=0; wm<wm_size; wm++) {
        int global_m = offset_m + local_m + wm*local_size_m;
        for (int wn=0; wn<wn_size; wn++) {
            int global_n = offset_n + local_n + wn*local_size_n;
            if (bias != NULL) {
                acc[wm][wn] += bias[global_n];
            }
            C[global_m*N + global_n] = acc[wm][wn];
        }
    }
}

__kernel void tile_linear(
    __global float *A,
    __global float *B,
    __global float *bias,
    __global float *C,
    const int K
) {

    const int local_i = get_local_size(0);
    const int local_j = get_local_size(1);

    int M = get_global_size(0);
    int N = get_global_size(1);

    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[i * K +  k] * B[j * K + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (bias != NULL) {
        sum += bias[j];
    }
    C[i * N + j] = sum;
}

__kernel void tile_reg_n_linear(
    __global float *A,
    __global float *B,
    __global float *bias,
    __global float *C,
    const int K
) {

    const int reg_size_n = 2;
    const int local_i = get_local_size(0);
    const int local_j = get_local_size(1);

    const int group_j = get_group_id(1);

    const int local_id_j = get_local_id(1);

    int M = get_global_size(0);
    int global_j = get_global_size(1);
    int N = global_j * reg_size_n;

    int i = get_global_id(0);
    int j = group_j * local_j * reg_size_n + local_id_j;

    float acc[reg_size_n];

    for (int wn=0; wn<reg_size_n; wn++) {
        acc[wn] = 0.0f;
    }

    for (int k = 0; k < K; k++) {
        for (int wn=0; wn<reg_size_n; wn++) {
            acc[wn] += A[i * K +  k] * B[(j + wn * local_j) * K + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (bias != NULL) {
        for (int wn=0; wn<reg_size_n; wn++) {
            acc[wn] += bias[j + wn * local_j];
        }
    }

    for (int wn=0; wn<reg_size_n; wn++) {
        C[i * N + j + wn * local_j] = acc[wn];
    }
}

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

__kernel void tile_reg_n_vector_linear(
    __global floatX *A,
    __global floatX *B,
    __global float *bias,
    __global float *C,
    const int K
) {

    const int reg_size_n = 8;
    const int local_i = get_local_size(0);
    const int local_j = get_local_size(1);

    const int group_j = get_group_id(1);

    const int local_id_j = get_local_id(1);

    int M = get_global_size(0);
    int global_j = get_global_size(1);
    int N = global_j * reg_size_n;

    int i = get_global_id(0);
    int j = group_j * local_j * reg_size_n + local_id_j;

    float acc[reg_size_n];

    for (int wn=0; wn<reg_size_n; wn++) {
        acc[wn] = 0.0f;
    }

    floatX vecA, vecB;
    const int K_div_width = K / WIDTH;
    for (int k = 0; k < K_div_width; k++) {
        vecA = A[i * K_div_width +  k];
        for (int wn=0; wn<reg_size_n; wn++) {
            vecB = B[(j + wn * local_j) * K_div_width + k];
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

    if (bias != NULL) {
        for (int wn=0; wn<reg_size_n; wn++) {
            acc[wn] += bias[j + wn * local_j];
        }
    }

    for (int wn=0; wn<reg_size_n; wn++) {
        C[i * N + j + wn * local_j] = acc[wn];
    }
}