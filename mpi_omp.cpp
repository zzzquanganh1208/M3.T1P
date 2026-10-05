#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    int size = argc > 1 ? stoi(argv[1]) : 500;
    int threadCount = argc > 2 ? stoi(argv[2]) : 2;

    omp_set_num_threads(threadCount);

    if (size % processCount != 0) {
        if (rank == 0) cout << "Matrix size must be divisible by number of processes." << endl;
        MPI_Finalize();
        return 0;
    }

    int numberOfLocalRows = size / processCount;

    vector<float> A, C;
    vector<float> B(size * size);
    vector<float> localRowsOfA(numberOfLocalRows * size);
    vector<float> localRowsOfC(numberOfLocalRows * size, 0);

    if (rank == 0) {
        A.assign(size * size, 0);
        B.assign(size * size, 1);
        C.resize(size * size);
        for (int i = 0; i < size; i++) A[i * size + i] = 1;
    }

    auto start = high_resolution_clock::now();

    MPI_Scatter(A.data(), numberOfLocalRows * size, MPI_FLOAT, localRowsOfA.data(), numberOfLocalRows * size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numberOfLocalRows; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                localRowsOfC[i * size + j] += localRowsOfA[i * size + k] * B[k * size + j];

    MPI_Gather(localRowsOfC.data(), numberOfLocalRows * size, MPI_FLOAT, C.data(), numberOfLocalRows * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto end = high_resolution_clock::now();

    double localTime = duration<double>(end - start).count();
    double totalTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool correct = true;
        for (float value : C) if (value != 1) correct = false;
        cout << "N = " << size << " | Processes = " << processCount << " | Threads = " << threadCount << " | Time = " << totalTime << " s | Correct = " << (correct ? "Yes" : "No") << endl;
    }

    MPI_Finalize();
    return 0;
}