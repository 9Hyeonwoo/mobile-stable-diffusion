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