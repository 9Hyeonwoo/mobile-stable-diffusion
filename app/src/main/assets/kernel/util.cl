__kernel void elemwise_add(__global float *A,
            __global float *B,
            __global float *C)
{

    // Get the work-item’s unique ID
    int idx = get_global_id(0);

    // Add the corresponding locations of
    // 'A' and 'B', and store the result in 'C'.
    C[idx] = A[idx] + B[idx];
}

__kernel void elemwise_multiply(__global float *A,
            __global float *B,
            __global float *C)
{

    // Get the work-item’s unique ID
    int idx = get_global_id(0);

    // Add the corresponding locations of
    // 'A' and 'B', and store the result in 'C'.
    C[idx] = A[idx] * B[idx];
}

__kernel void permute3D__1_0_2(__global float *src,
                      __global float *dst)
{
    // i j k -> j i k.
    int iSize = get_global_size(0);
    int jSize = get_global_size(1);
    int kSize = get_global_size(2);

    int offset = get_global_offset(0) * jSize * kSize + get_global_offset(1) * kSize + get_global_offset(2);

    int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int j = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int k = get_group_id(2) * get_local_size(2) + get_local_id(2);

    int src_idx = i * jSize * kSize + j * kSize + k + offset;
    int dst_idx = j * iSize * kSize + i * kSize + k + offset;
    dst[dst_idx] = src[src_idx];
}


__kernel void permute3D__0_2_1(__global float *src,
                      __global float *dst)
{
    // i j k -> i k j.
    int iSize = get_global_size(0);
    int jSize = get_global_size(1);
    int kSize = get_global_size(2);

    int offset = get_global_offset(0) * jSize * kSize + get_global_offset(1) * kSize + get_global_offset(2);

    int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int j = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int k = get_group_id(2) * get_local_size(2) + get_local_id(2);

    int src_idx = i * jSize * kSize + j * kSize + k + offset;
    int dst_idx = i * kSize * jSize + k * jSize + j + offset;
    dst[dst_idx] = src[src_idx];
}

__kernel void permute3D(
    __global float *src,
    __global float *dst,
    const int dst_first_dim,
    const int dst_second_dim,
    const int dst_third_dim
) {
     int permute[3] = {dst_first_dim, dst_second_dim, dst_third_dim};
    int iSize = get_global_size(0);
    int jSize = get_global_size(1);
    int kSize = get_global_size(2);

    int offset = get_global_offset(0) * jSize * kSize + get_global_offset(1) * kSize + get_global_offset(2);

    int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int j = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int k = get_group_id(2) * get_local_size(2) + get_local_id(2);

    int src_idx = i * jSize * kSize + j * kSize + k + offset;

    int dst_idx = 0;
    for (int index = 0; index < 3; index++) {
        if (permute[index] == 0) {
            dst_idx *=iSize;
            dst_idx += i;
        } else if (permute[index] == 1) {
            dst_idx *= jSize;
            dst_idx += j;
        } else if (permute[index] == 2) {
            dst_idx *= kSize;
            dst_idx += k;
        }
    }
    dst_idx += offset;

    dst[dst_idx] = src[src_idx];
}

__kernel void gelu(__global float *src,
                   __global float *dst)
{
    int idx = get_global_id(0);

    // GELU(x) = x * 0.5 * (1.0 + erf( x / \sqrt{2}))
    // tanh version
    // const float x = src[idx];
    // float temp = pown(x, 3);
    // temp = fma(temp, 0.044715f, x);
    // temp *= sqrt(M_2_PI_F);
    // temp = tanh(temp) + 1.0f;
    // dst[idx] = 0.5f * x * temp;

    const float x = src[idx];
    float temp = x * M_SQRT1_2_F;
    temp = 1.0f + erf(temp);
    dst[idx] = x * 0.5f * temp;
}

__kernel void silu(__global float *src,
                   __global float *dst)
{
    int idx = get_global_id(0);

    // SILU(x) = x / (1.0 + exp(-x))
    const float x = src[idx];
    float temp = exp(-x);
    temp = 1.0f / (1.0f + temp);
    dst[idx] = x * temp;
}

__kernel void chunkwise_add(__global float *A,
            __global float *chunk,
            __global float *C,
            const size_t chunk_size)
{
    // Get the work-item’s unique ID
    int idx = get_global_id(0);
    int chunk_idx = idx / chunk_size;

    // Add the corresponding locations of
    // 'A' and 'chunk', and store the result in 'C'.
    C[idx] = A[idx] + chunk[chunk_idx];
}

__kernel void softmax(
    __global const float *input,
    __global float *output,
    __local float* reductionSums,
    __local float* cache,
    const size_t chunkSize
) {
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = get_group_id(0);
	const int globalOffset = workgroupID * chunkSize;

    const int i = globalOffset + localID;
    float sum = 0.0f;
    for(int index = i; index < (globalOffset + chunkSize); index += localSize) {
        float expVal = exp(input[index]);
        sum += expVal;
        // save to cache using adjacent memory locations
        cache[index - globalOffset] = expVal;
    }
	reductionSums[localID] = sum;

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

    for(int index = i; index < (globalOffset + chunkSize); index += localSize) {
        output[index] = cache[index - globalOffset] / reductionSums[0];
    }
}

__kernel void batch_matmul(__global float *A,
                        __global float *B,
                        __global float *output,
                        const size_t K,
                        const float scale)
{
    int batch_size = get_global_size(0);
    int M = get_global_size(1);
    int N = get_global_size(2);

    int batch = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A(M, K) * B(K, N) = C(M, N)
        sum += A[batch * M * K + i * K + k] * B[batch * K * N + k * N + j];
    }
    output[batch * M * N + i * N + j] = sum * scale;
}

__kernel void batch_matmul_scale(
    __global const float *A, // A = (M, K)
    __global const float *B, // B = (K, N)
    __global float* C,
    const size_t M,
    const size_t N,
    const size_t K,
    const float scale
) {

    const int reg_size_m = 8;
    const int tile_size_m = 128;
    const int reg_size_n = 8;
    const int tile_size_n = 128;
    const int tile_size_k = 16;

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
        // __global const float* Asub_global = A + offset_m * K + t * tile_size_k + offset_batch_A;
        // __global const float* Bsub_global = B + (t * tile_size_k) * N + offset_n + offset_batch_B;

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
            C[global_m*N + global_n + offset_batch_C] = acc[wm][wn] * scale;
        }
    }
}