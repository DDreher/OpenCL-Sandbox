#pragma once
#include <vector>
#include <CL\cl_platform.h>

class PrefixSum
{
public:
    static std::vector<cl_int> CalculateCPU(const std::vector<cl_int>& elements);
    static std::vector<cl_int> CalculateGPU(const std::vector<cl_int>& elements);
    static int praefixsumme(cl_int* input, cl_int* output, int size);
   
private:
    static std::vector<cl_int> CalculateGPU_256(const std::vector<cl_int>& elements);
};
