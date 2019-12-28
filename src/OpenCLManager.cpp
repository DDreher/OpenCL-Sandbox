#include "OpenCLManager.h"
#include "Definitions.h"
#include <assert.h>

OpenCLManager::OpenCLManager()
{
    Init();
}

OpenCLManager::~OpenCLManager()
{
}

ReturnCode OpenCLManager::Init()
{
    cl_uint device = 1;
    cl_uint num_platforms = 0;
    cl_platform_id platform = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);

    assert(status != ReturnCode::ERROR);

    return ReturnCode::SUCCESS;
}
