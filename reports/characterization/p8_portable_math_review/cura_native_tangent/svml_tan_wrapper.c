#include <immintrin.h>
#include <stddef.h>

extern __m512 __svml_tanf16(__m512);

void svml_tanf_array(const float *input, float *output, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 value = _mm512_loadu_ps(input + i);
        _mm512_storeu_ps(output + i, __svml_tanf16(value));
    }
    if (i < count) {
        float tail_in[16] = {0};
        float tail_out[16];
        for (size_t j = 0; i + j < count; ++j) tail_in[j] = input[i + j];
        _mm512_storeu_ps(tail_out, __svml_tanf16(_mm512_loadu_ps(tail_in)));
        for (size_t j = 0; i + j < count; ++j) output[i + j] = tail_out[j];
    }
}
