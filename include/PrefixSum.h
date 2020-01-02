#pragma once
#include <vector>
#include <CL\cl.h>

class PrefixSum
{
public:
    static std::vector<cl_int> CalculateCPU(const std::vector<cl_int>& elements);
    static std::vector<cl_int> CalculateGPU(const std::vector<cl_int>& elements);
    static int praefixsumme(cl_int* input, cl_int* output, int size);
   
private:
    static std::vector<cl_int> CalculateGPU_Recursive(cl_mem a_buffer, cl_mem b_buffer, size_t num_elements);
};
