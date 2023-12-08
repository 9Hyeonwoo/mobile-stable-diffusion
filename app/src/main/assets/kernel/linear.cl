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