#include "Rand.h"

void Rand(float * pArray, int n, int x)
{
    srand(108719 * x + 1);

    for (int ip = 0; ip < n; ip++)
    {
        pArray[ip] = rand() % 1025 - 512;
    }
}