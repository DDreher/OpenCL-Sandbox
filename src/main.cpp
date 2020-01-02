#include <iostream>
#include <filesystem>
#include "OpenCLManager.h"
#include "PrefixSum.h"
#include "Definitions.h"

int main(int, char**)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel(filenames::KERNELS_PREFIX_SUM, { kernels::PREFIX_SUM });

    std::vector<cl_int> test_elements;
    std::vector<cl_int> expected_output;
    
    uint32_t tmp_num = 0;
    for (uint32_t i = 0; i < 256; ++i)
    {
        test_elements.push_back(i+1);
        expected_output.push_back(tmp_num);
        tmp_num += i+1;
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
    
    /*int size = 256;
    cl_int* input = new cl_int[size];
    cl_int* output = new cl_int[size];

    for (int i = 0; i < size; i++)
        input[i] = 1;

    PrefixSum::praefixsumme(input, output, size);
    memcmp(input, output, size);
    if (memcmp(input, output, size))
    {
        std::cout << "True" << std::endl;
    }
    else
    {
        std::cout << "False" << std::endl;
    }*/

    return 0;
}
