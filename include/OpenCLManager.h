#pragma once
#include "Definitions.h"

#include <CL/cl.h>
#include <string>
#include <unordered_map>

class OpenCLManager
{
public:
    OpenCLManager();
    ~OpenCLManager();

    cl_context context = 0;
    cl_command_queue command_queue = 0;
    cl_program program = 0;

    void LoadKernel(const std::string& file_name, std::initializer_list<std::string> kernel_names);

    std::unordered_map<std::string, cl_kernel> kernel_map;

private:
    void Init();

    cl_device_id device_id_ = 0;
};
