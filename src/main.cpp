#include <iostream>
#include <filesystem>
#include "OpenCLManager.h"
#include "PrefixSum.h"

int main(int, char**)
{
    OpenCLManager* mgr = OpenCLManager::GetInstance();
    mgr->LoadKernel("kernel_prefix_sum.cl", { "PrefixSum" });

    std::vector<cl_int> test_elements = { 3,2,1,2,1,4,3,2 };
    std::vector<cl_int> expected_output = { 0,3,5,6,8,9,13,16 };
    std::vector<cl_int> test_prefix_sum = PrefixSum::CalculateCPU(test_elements);

    if (test_prefix_sum == expected_output)
    {
        std::cout << "True" << std::endl;
    }
    else
    {
        std::cout << "False" << std::endl;
    }

    return 0;
}
