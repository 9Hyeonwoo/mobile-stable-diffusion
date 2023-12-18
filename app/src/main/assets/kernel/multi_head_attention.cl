__kernel void add_matmul_attention(__global float *QKV,
                        __global float *attn_mask,
                        __global float *output,
                        const size_t K)
{
    // QKV: (3, NUM_HEADS(16), CONTEXT_LENGTH(77), 64)
    int batch_size = get_global_size(0);
    int M = get_global_size(1);
    int N = get_global_size(2);

    int batch = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        int q_idx = batch * M * K + i * K + k;
        int k_idx = batch * N * K + j * K + k + (batch_size * M * K);
        sum += QKV[q_idx] * QKV[k_idx];
    }

    int output_idx = batch * M * N + i * N + j;
    int attn_mask_idx = i * N + j;
    output[output_idx] = sum / sqrt((float)K) + attn_mask[attn_mask_idx];
}

__kernel void local_softmax(__global const float *input,
                        __global float *output,
                        __local float* reductionSums)
{
    const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = get_group_id(0);

	const float expVal = exp(input[globalID]);

	reductionSums[localID] = expVal;

	int remainder = localSize % 2;
	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID + remainder] += reductionSums[localID + offset + remainder];
		}
	    offset += remainder;
	    remainder = offset % 2;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

	output[globalID] = expVal / reductionSums[0];
}

__kernel void batch_matmul_attention(__global float *QK,
                        __global float *QKV,
                        __global float *output,
                        const size_t K)
{
    int batch_size = get_global_size(0);
    int M = get_global_size(1);
    int N = get_global_size(2);

    int batch = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    float sum = 0.0f;
    int V_offset = batch_size * K * N * 2;
    for (int k = 0; k < K; k++) {
        // A(M, K) * B(K, N) = C(M, N)
        sum += QK[batch * M * K + i * K + k] * QKV[batch * K * N + k * N + j + V_offset];
    }
    output[batch * M * N + i * N + j] = sum;
}

__kernel void batch_matmul_mask(
    __global const float *A, // A = (M, K)
    __global const float *B, // B = (N, K)
    __global const float *attn_mask,
    __global float* C,
    const size_t M,
    const size_t N,
    const size_t K
    // const size_t reg_size_n,
    // const size_t tile_size_n,
    // const size_t tile_size_k
) {

    const int reg_size_m = 7;
    const int tile_size_m = 77;
    const int reg_size_n = 7;
    const int tile_size_n = 77;
    const int tile_size_k = 16;

    // Thread identifiers
    const int offset_batch_A = get_global_id(0) * M * K ;
    const int offset_batch_B = get_global_id(0) * N * K ;
    const int offset_batch_C = get_global_id(0) * M * N ;
    const int local_size_m = get_local_size(1);
    const int local_size_n = get_local_size(2);
    const int local_m = get_local_id(1);
    const int local_n = get_local_id(2);
    const int offset_m = get_group_id(1) * get_local_size(1) * reg_size_m ;
    const int offset_n = get_group_id(2) * get_local_size(2) * reg_size_n ;

    // Local memory to fit a tile of A and B
    __local float Asub[tile_size_m * tile_size_k];
    __local float Bsub[tile_size_k * tile_size_n];

    // Allocate register space
    float Areg;
    float Breg[reg_size_n];
    float acc[reg_size_m][reg_size_n];

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
        // Load one tile of A and B into local memory
        for (int id=local_m*local_size_n + local_n; id<(tile_size_m * tile_size_k); id+=local_size_m*local_size_n) {
            int m = id / tile_size_k;
            int k = id % tile_size_k;
            int tiledIndex = tile_size_k*t + k;
            // A = (M, K), B = (K, N)
            int A_index = (offset_m + m)*K + (tiledIndex);
            if (A_index < M*K) {
                Asub[m*tile_size_k + k] = A[A_index + offset_batch_A];
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
                Bsub[k*tile_size_n + n] = B[B_index + offset_batch_B];
            } else {
                break;
            }
        }
        // global version
        // __global const float* Asub_global = A + offset_m * K + t * tile_size_k + offset_batch_A;
        // __global const float* Bsub_global = B + offset_n * K + t * tile_size_k + offset_batch_B;

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
            C[global_m*N + global_n + offset_batch_C] = acc[wm][wn] / sqrt((float)K) + attn_mask[global_m*N + global_n];
        }
    }
}

__kernel void batch_matmul(
    __global const float *A, // A = (M, K)
    __global const float *B, // B = (K, N)
    __global float* C,
    const size_t M,
    const size_t N,
    const size_t K
) {

    const int reg_size_m = 7;
    const int tile_size_m = 77;
    const int reg_size_n = 8;
    const int tile_size_n = 64;
    const int tile_size_k = 11;

    // Thread identifiers
    const int offset_batch_A = get_global_id(0) * M * K ;
    const int offset_batch_B = get_global_id(0) * K * N ;
    const int offset_batch_C = get_global_id(0) * M * N ;
    const int local_size_m = get_local_size(1);
    const int local_size_n = get_local_size(2);
    const int local_m = get_local_id(1);
    const int local_n = get_local_id(2);
    const int offset_m = get_group_id(1) * get_local_size(1) * reg_size_m ;
    const int offset_n = get_group_id(2) * get_local_size(2) * reg_size_n ;

    // Local memory to fit a tile of A and B
    __local float Asub[tile_size_m * tile_size_k];
    __local float Bsub[tile_size_k * tile_size_n];

    // Allocate register space
    float Areg;
    float Breg[reg_size_n];
    float acc[reg_size_m][reg_size_n];

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

        // Load one tile of A and B into local memory
        for (int id=local_m*local_size_n + local_n; id<(tile_size_m * tile_size_k); id+=local_size_m*local_size_n) {
            int m = id / tile_size_k;
            int k = id % tile_size_k;
            int tiledIndex = tile_size_k*t + k;
            // A = (M, K), B = (K, N)
            int A_index = (offset_m + m)*K + (tiledIndex);
            if (A_index < M*K) {
                Asub[m*tile_size_k + k] = A[A_index + offset_batch_A];
            } else {
                break;
            }
        }
        for (int id=local_m*local_size_n + local_n; id<(tile_size_n * tile_size_k); id+=local_size_m*local_size_n) {
            int n = id / tile_size_k;
            int k = id % tile_size_k;
            int tiledIndex = tile_size_k*t + k;
            // A = (M, K), B = (K, N)
            int B_index = (tiledIndex)*N + (offset_n + n);
            if (B_index < N*K) {
                Bsub[k*tile_size_n + n] = B[B_index + offset_batch_B];
            } else {
                break;
            }
        }
        // global version
        //__global const float* Asub_global = A + offset_m * K + t * tile_size_k + offset_batch_A;
        //__global const float* Bsub_global = B + (t * tile_size_k) * N + offset_n + offset_batch_B;

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<tile_size_k; k++) {

            // Cache the values of Bsub in registers
            for (int wn=0; wn<wn_size; wn++) {
                int col = local_n + wn*local_size_n;
                Breg[wn] = Bsub[k*tile_size_n + col];
                //Breg[wn] = Bsub_global[k * N + col];
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
            C[global_m*N + global_n + offset_batch_C] = acc[wm][wn];
        }
    }
}