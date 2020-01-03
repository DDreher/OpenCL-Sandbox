#pragma once

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
        static constexpr size_t MAX_WORK_GROUP_SIZE = 256;   // Just an assumption for academic purposes. Real value depends on device!
    };

    enum ReturnCode
    {
        CODE_SUCCESS = 0,
        CODE_ERROR = 1
    };
};
