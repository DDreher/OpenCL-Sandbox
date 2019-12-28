#pragma once
#include <vector>
#include <CL\cl_platform.h>

class PrefixSum
{
public:
    static std::vector<cl_int> CalculateCPU(const std::vector<cl_int>& elements);
};