#include "PrefixSum.h"

std::vector<cl_int> PrefixSum::CalculateCPU(const std::vector<cl_int>& elements)
{
    uint32_t sum = 0;

    std::vector<cl_int> prefix_sum;
    prefix_sum.reserve(elements.size());

    for(cl_int element : elements)
    {
        prefix_sum.push_back(sum);
        sum += element;
    }

    return prefix_sum;
}
