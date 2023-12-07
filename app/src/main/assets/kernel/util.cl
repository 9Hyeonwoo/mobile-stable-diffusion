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
