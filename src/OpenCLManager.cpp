#include "OpenCLManager.h"
#include "Definitions.h"
#include "Utilities.h"
#include <assert.h>
#include <iostream>

OpenCLManager* OpenCLManager::instance_ = nullptr;

OpenCLManager::OpenCLManager()
{
    Init();
}

OpenCLManager::~OpenCLManager()
{
    // Release kernels
    for (auto& [kernel_name, kernel] : kernel_map)
    {
        clReleaseKernel(kernel);
    }

    // Release cl objects
    if (program != 0)
    {
        clReleaseProgram(program);
    }

    if (command_queue != 0)
    {
        clReleaseCommandQueue(command_queue);
    }

    if (context != 0)
    {
        clReleaseContext(context);
    }
}

OpenCLManager* OpenCLManager::GetInstance()
{
    if(instance_ == nullptr)
    {
        instance_ = new OpenCLManager();
    }

    return instance_;
}

void OpenCLManager::TearDown()
{
    delete instance_;
    instance_ = nullptr;
}

void OpenCLManager::LoadKernel(const std::string& file_name, std::initializer_list<std::string> kernel_names)
{
    cl_int status = 0;

    // Read file content
    auto [return_code, file_content] = Utility::ReadFile("src/kernels/" + file_name);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // Create program
    const char* program_source = file_content.c_str();
    size_t source_length = strlen(file_content.c_str());
    program = clCreateProgramWithSource(context, 1, &program_source, &source_length, &status);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // Compile program
    status = clBuildProgram(program, 1, &device_id_, NULL, NULL, NULL);

    // Check compilation status log
    if (status != mpp::ReturnCode::CODE_SUCCESS)
    {
        char msg[120000];
        clGetProgramBuildInfo(program, device_id_, CL_PROGRAM_BUILD_LOG, sizeof(msg), msg, NULL);
        std::cerr << "=== build failed ===\n" << msg << std::endl;
        getc(stdin);
    }
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // Create kernel objects
    for (auto& kernel_name : kernel_names)
    {
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &status);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);

        kernel_map[kernel_name] = kernel;
    }
}

void OpenCLManager::Init()
{
    cl_int status = 0;

    // Choose first available platform
    cl_uint num_platforms = 0;
    status = clGetPlatformIDs(0, nullptr, &num_platforms);
    assert(status != mpp::ReturnCode::CODE_ERROR);
    assert(num_platforms > 0);

    cl_platform_id* available_platforms = new cl_platform_id[num_platforms];
    status = clGetPlatformIDs(num_platforms, available_platforms, nullptr);

    cl_platform_id platform = available_platforms[0];
    delete[] available_platforms;
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    // Set device (prefer GPU over CPU)
    cl_uint num_devices = 0;
    cl_device_id* devices = nullptr;
    cl_device_type chosen_device_type = CL_DEVICE_TYPE_GPU;
    status = clGetDeviceIDs(platform, chosen_device_type, 0, nullptr, &num_devices);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    if (num_devices == 0)	// no GPU available
    {
        std::cout << "No GPU device available." << std::endl << "Setting CPU as default device." << std::endl;
        chosen_device_type = CL_DEVICE_TYPE_CPU;
        status = clGetDeviceIDs(platform, chosen_device_type, 0, nullptr, &num_devices);
        assert(status == mpp::ReturnCode::CODE_SUCCESS);
        assert(num_devices > 0);
    }

    devices = new cl_device_id[num_devices];
    status = clGetDeviceIDs(platform, chosen_device_type, num_devices, devices, nullptr);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
    assert(devices != nullptr);
    device_id_ = devices[0];
    delete[] devices;

    // Set up cl context and command queue
    context = clCreateContext(nullptr, 1, &device_id_, nullptr, nullptr, &status);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);

    command_queue = clCreateCommandQueue(context, device_id_, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
    assert(status == mpp::ReturnCode::CODE_SUCCESS);
}
