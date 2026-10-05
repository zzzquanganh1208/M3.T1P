#include <mpi.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

const char* kernelSource = R"(
__kernel void multiply(__global float* A, __global float* B, __global float* C, int localRows, int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < localRows && col < size) {
        float sum = 0;

        for (int k = 0; k < size; k++)
            sum += A[row * size + k] * B[k * size + col];

        C[row * size + col] = sum;
    }
}
)";

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    int size = argc > 1 ? stoi(argv[1]) : 500;

    if (size % processCount != 0) {
        if (rank == 0) cout << "Matrix size must be divisible by number of processes." << endl;
        MPI_Finalize();
        return 0;
    }

    int numberOfLocalRows = size / processCount;

    vector<float> A, C;
    vector<float> B(size * size);
    vector<float> localRowsOfA(numberOfLocalRows * size);
    vector<float> localRowsOfC(numberOfLocalRows * size);

    if (rank == 0) {
        A.assign(size * size, 0);
        B.assign(size * size, 1);
        C.resize(size * size);
        for (int i = 0; i < size; i++) A[i * size + i] = 1;
    }

    auto start = high_resolution_clock::now();

    MPI_Scatter(A.data(), numberOfLocalRows * size, MPI_FLOAT, localRowsOfA.data(), numberOfLocalRows * size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    cl_platform_id platform;
    cl_device_id device;
    cl_int error;

    clGetPlatformIDs(1, &platform, nullptr);

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (error != CL_SUCCESS){
        cout << "Rank " << rank << ": GPU not found. Using CPU instead." << endl;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }

    if (error != CL_SUCCESS) {
        cout << "Rank " << rank << ": No OpenCL device found." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &error);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiply", &error);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, localRowsOfA.size() * sizeof(float), localRowsOfA.data(), &error);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), B.data(), &error);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, localRowsOfC.size() * sizeof(float), nullptr, &error);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &numberOfLocalRows);
    clSetKernelArg(kernel, 4, sizeof(int), &size);

    size_t globalSize[2] = {(size_t)numberOfLocalRows, (size_t)size};

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, localRowsOfC.size() * sizeof(float), localRowsOfC.data(), 0, nullptr, nullptr);

    MPI_Gather(localRowsOfC.data(), numberOfLocalRows * size, MPI_FLOAT, C.data(), numberOfLocalRows * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto end = high_resolution_clock::now();

    double localTime = duration<double>(end - start).count();
    double totalTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool correct = true;
        for (float value : C) if (value != 1) correct = false;

        cout << "N = " << size << " | Processes = " << processCount << " | Time = " << totalTime << " s | Correct = " << (correct ? "Yes" : "No") << endl;
    }

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    MPI_Finalize();
    return 0;
}