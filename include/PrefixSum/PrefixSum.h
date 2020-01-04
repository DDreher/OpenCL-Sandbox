#pragma once
#include <vector>
#include <CL\cl.h>

class PrefixSum
{
public:
    static std::vector<cl_int> CalculateCPU(const std::vector<cl_int>& elements);
    static std::vector<cl_int> CalculateGPU(const std::vector<cl_int>& elements);
   
private:
    static void CalculateGPU_Recursive(cl_mem a_buffer, cl_mem b_buffer, size_t num_elements);
};
