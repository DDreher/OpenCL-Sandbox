#include <iostream>
#include <filesystem>
#include "OpenCLManager.h"
#include "PrefixSum.h"
#include "Definitions.h"

int main(int, char**)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(filenames::KERNELS_PREFIX_SUM, { kernels::PREFIX_SUM, kernels::PREFIX_CALC_E });

    std::vector<cl_int> test_elements;
    std::vector<cl_int> expected_output;
    
    cl_int tmp_num = 0;
    for (cl_int i = 0; i < 256; ++i)
    {
        cl_int val = 1;
        test_elements.push_back(val);
        expected_output.push_back(tmp_num);
        tmp_num += val;
    }

    std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);
    std::cout << "Prefix sum CPU: ";
    if (test_prefix_sum == expected_output)
    {
        std::cout << "True" << std::endl;
    }
    else
    {
        std::cout << "False" << std::endl;
    }

    test_prefix_sum = PrefixSum::CalculateGPU(test_elements);
    std::cout << "Prefix sum GPU: ";
    if (test_elements == test_prefix_sum)
    {
        std::cout << "True" << std::endl;
    }
    else
    {
        std::cout << "False" << std::endl;
    }

    return 0;
}
