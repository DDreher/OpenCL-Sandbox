
# Overview

This repository contains an OpenCL implementation of both a Prefix Scan and a Cuckoo Hash. The solution is split up in corresponding projects:

## Base

Contains the foundation for the OpenCL implementations and their tests. Most importantly it contains the OpenCLManager which is in charge of managing the OpenCL context, the programs and the kernels.

# PrefixScan

Contains a Blelloch Scan implementation for both host and device for comparison. The calculation is done recursively.

Reference: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

# PrefixScan_Test

Contains test and performance comparisons of the PrefixScan algorithms. Tests are implemented with Catch22.

# HashTable

Contains a Cuckoo Hash implementation for the device.

Reference: https://www.researchgate.net/publication/211178395_Building_an_Efficient_Hash_Table_on_the_GPU

# HashTable_Test

Contains test and performance comparisons of the Cuckoo Hash implementation. For comparison with the host a std::unordered_map is used. Tests are implemented with Catch22.