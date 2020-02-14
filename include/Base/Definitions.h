#pragma once
#include <array>
#include <CL/cl.h>

namespace mpp
{
    namespace filenames
    {
        static constexpr char KERNELS_PREFIX_SUM[] = "kernel_prefix_sum.cl";
        static constexpr char KERNELS_HASHTABLE[] = "kernel_hashtable.cl";
    };

    namespace kernels
    {
        static constexpr char PREFIX_SUM[] = "PrefixSum256";
        static constexpr char PREFIX_CALC_E[] = "CalcE";

        static constexpr char HASHTABLE_INSERT[] = "Insert";
        static constexpr char HASHTABLE_RETRIEVE[] = "Retrieve";
    };

    namespace constants
    {
        static constexpr size_t MAX_THREADS_PER_CU = 256;   // Just an assumption for academic purposes. Real value depends on device!
        static constexpr std::array<cl_int, MAX_THREADS_PER_CU> ZEROS = std::array<cl_int, MAX_THREADS_PER_CU>();
        static constexpr size_t WAVEFRONT_SIZE = 32;
        static constexpr uint64_t EMPTY = -1;
    };

    enum ReturnCode
    {
        CODE_SUCCESS = 0,
        CODE_ERROR = 1
    };
};
