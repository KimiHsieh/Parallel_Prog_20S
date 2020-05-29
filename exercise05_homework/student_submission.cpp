#include "dgemm.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <inttypes.h>

// float sum8(__m256 x){
//     __m128 x0 = _mm256_castps256_ps128(x);
//     __m128 x1 = _mm256_extractf128_ps(x, 1);
//     __m128 y = _mm_add_ps(x0, x1);
//     __m128 y0 = y;
//     __m128 y1 = _mm_movehl_ps(y, y);
//     __m128 z = _mm_add_ps(y0, y1);
//     __m128 z0 = z;
//     __m128 z1 = _mm_shuffle_ps(z, z, 0x1);
//     __m128 w = _mm_add_ss(z0, z1);
//     return _mm_cvtss_f32(w);
// }

void dgemm(float alpha, const float *a, const float *b, float beta, float *c)
{
    // TODO: insert your solution here
    int vec_length = 8;
    int vec_end = MATRIX_SIZE - (MATRIX_SIZE % vec_length);
    float sum;
    __m256 A, B, sum_v, mul_v;

    for (int i = 0; i < MATRIX_SIZE; i++) // row index
    {
        __m256i mask = _mm256_setr_epi32(
            -(0 < MATRIX_SIZE % 8), -(1 < MATRIX_SIZE % 8), -(2 < MATRIX_SIZE % 8), -(3 < MATRIX_SIZE % 8),
            -(4 < MATRIX_SIZE % 8), -(5 < MATRIX_SIZE % 8), -(6 < MATRIX_SIZE % 8), -(7 < MATRIX_SIZE % 8));

        for (int j = 0; j < MATRIX_SIZE; j++) //column index
        {
            sum = 0;
            sum_v = _mm256_set1_ps(0.f);
            c[i * MATRIX_SIZE + j] *= beta;
            int k = 0;
            for (; k + vec_length <= MATRIX_SIZE; k += vec_length)
            {
                A = _mm256_loadu_ps(a + i * MATRIX_SIZE + k);
                B = _mm256_loadu_ps(b + j * MATRIX_SIZE + k);
                mul_v = _mm256_mul_ps(A, B);
                sum_v = _mm256_add_ps(sum_v, mul_v);
                // _mm256_storeu_ps(c[i], mul_v);
            }
            // Process remainder
            A = _mm256_maskload_ps(a + i * MATRIX_SIZE + k, mask);
            B = _mm256_maskload_ps(b + j * MATRIX_SIZE + k, mask);
            mul_v = _mm256_mul_ps(A, B);
            sum_v = _mm256_add_ps(sum_v, mul_v);

            for (int l = 0; l < vec_length; l++)
            {
                sum += sum_v[l];
            }

            c[i * MATRIX_SIZE + j] += alpha * sum;
        }
    }
}

int main(int, char **)
{
    float alpha, beta;

    // mem allocations
    int mem_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    auto a = (float *)malloc(mem_size);
    auto b = (float *)malloc(mem_size);
    auto c = (float *)malloc(mem_size);

    // check if allocated
    if (nullptr == a || nullptr == b || nullptr == c)
    {
        printf("Memory allocation failed\n");
        if (nullptr != a)
            free(a);
        if (nullptr != b)
            free(b);
        if (nullptr != c)
            free(c);
        return 0;
    }

    generateProblemFromInput(alpha, a, b, beta, c);

    std::cerr << "Launching dgemm step." << std::endl;
    // matrix-multiplication
    dgemm(alpha, a, b, beta, c);

    outputSolution(c);

    free(a);
    free(b);
    free(c);
    return 0;
}
