
# Overview

This repository represents my personal Sandbox to play around with heterogenous parallel computing algorithms and techniques.
Right now it contains an OpenCL implementation of both a Prefix Scan and a Cuckoo Hash.

**Disclaimer: The implementations are for personal educational purposes only. It's not adviced to use them in any production environment.**

# How to build

The solution's paths are already set up relative to environment variables.    
The development environment relies on NVidia GPUs, therefore the OpenCL path depends on $(CUDA_PATH).    
If you use an AMD GPU or don't have this environment variable set you have to adjust the build properties of each project.

**IMPORTANT:**    
If you don't want to copy the kernel implementations (.cl files) into the build directory to debug / execute the unit tests it's recommended to set the 
working directory in the build properties of the Visual Studio projects under Configuration Properties -> Debugging -> Working Directory to $(ProjectDir).
Sadly this seems to be a local setting and can't be commited to the repository.

# Project Structure
The solution is split up in corresponding projects

## Base

Contains the foundation for the OpenCL implementations and their tests. Most importantly it contains the OpenCLManager which is in charge of managing the OpenCL context, the programs and the kernels.

## PrefixScan

Contains a Blelloch Scan implementation for both host and device for comparison. The calculation is done recursively.

**Reference:**    
https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

## PrefixScan_Test

Contains test and performance comparisons of the PrefixScan algorithms.

## HashTable

Contains a Cuckoo Hash implementation for the device.

**Reference:**    
https://www.researchgate.net/publication/211178395_Building_an_Efficient_Hash_Table_on_the_GPU

## HashTable_Test

Contains test and performance comparisons of the Cuckoo Hash implementation. For comparison with the host a std::unordered_map is used.

# Third Party Dependencies

Catch2 - https://github.com/catchorg/Catch2
