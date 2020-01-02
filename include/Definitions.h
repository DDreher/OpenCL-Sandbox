#pragma once

namespace filenames
{
    constexpr static const char* KERNELS_PREFIX_SUM = "kernel_prefix_sum.cl";
}

namespace kernels
{
    constexpr static const char* PREFIX_SUM = "PrefixSum_256";
}

namespace constants
{
    constexpr uint32_t MAX_FIELD_SIZE = 256;
}

enum ReturnCode
{
    SUCCESS = 0,
    ERROR = 1
};
