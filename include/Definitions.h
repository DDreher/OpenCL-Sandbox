#pragma once
#include <array>
#include <CL/cl.h>

namespace mpp
{
    namespace filenames
    {
        static constexpr char KERNELS_PREFIX_SUM[] = "kernel_prefix_sum.cl";
    };

    namespace kernels
    {
        static constexpr char PREFIX_SUM[] = "PrefixSum256";
        static constexpr char PREFIX_CALC_E[] = "CalcE";
    };

    namespace constants
    {
        static constexpr size_t MAX_THREADS_PER_CU = 256;   // Just an assumption for academic purposes. Real value depends on device!
        static constexpr std::array<cl_int, MAX_THREADS_PER_CU> ZEROS = std::array<cl_int, MAX_THREADS_PER_CU>();
    };

    enum ReturnCode
    {
        CODE_SUCCESS = 0,
        CODE_ERROR = 1
    };
};
