__kernel void linear(__global float *A,
                    __global float *B,
                    __global float *bias,
                    __global float *C,
                    const int M, const int N, const int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A(M, K) * B(N, K) = C(M, N)
        sum += A[i * K + k] * B[j * K + k];
    }
    C[i * N + j] = sum + bias[j];
}