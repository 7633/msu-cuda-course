#include <xmmintrin.h>
#include <stdio.h>

struct vec4
{
    union
    {
        float   v[4];
        __m128  v4;
    };
};

int main()
{
    vec4 a = {5.0f, 2.0f, 1.0f, 3.0f};
    vec4 b = {5.0f, 3.0f, 9.0f, 7.0f};
    vec4 c;

    c.v4 = _mm_add_ps(a.v4, b.v4);

    printf("c = {%.3f, %.3f, %.3f, %.3f}\n", c.v[0], c.v[1], c.v[2], c.v[3]);

    return 0;
}