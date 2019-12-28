#pragma once
#include "Definitions.h"

#include <CL/cl.h>

class OpenCLManager
{
public:
    OpenCLManager();
    ~OpenCLManager();

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;

private:
    ReturnCode Init();
};
