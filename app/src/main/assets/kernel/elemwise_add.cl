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