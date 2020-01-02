#pragma once

namespace filenames
{
    constexpr static const char* KERNELS_PREFIX_SUM = "kernel_prefix_sum.cl";
}

namespace kernels
{
    constexpr static const char* PREFIX_SUM = "PrefixSum256";
    constexpr static const char* PREFIX_CALC_E = "CalcE";
}

namespace constants
{
    constexpr static const size_t MAX_WORK_GROUP_SIZE = 256;   // Just an assumption for academic purposes. Real value depends on device!
}

enum ReturnCode
{
    SUCCESS = 0,
    ERROR = 1
};
