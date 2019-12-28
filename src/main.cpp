#include <iostream>
#include <filesystem>
#include "OpenCLManager.h"

int main(int, char**)
{
    OpenCLManager opencl_mgr;

    opencl_mgr.LoadKernel("sum_kernel.cl", { "summe_kernel" });
    return 0;
}
