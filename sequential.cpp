#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    int size = argc > 1 ? stoi(argv[1]) : 500;
    vector<float> A(size * size, 0), B(size * size, 1), C(size * size, 0);
    for (int i = 0; i < size; i++) A[i * size + i] = 1;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i * size + j] += A[i * size + k] * B[k * size + j];

    auto end = high_resolution_clock::now();

    bool correct = true;
    for (float value : C) if (value != 1) correct = false;
    cout << "N = " << size << " | Time = " << duration<double>(end - start).count() << " s | Correct = " << (correct ? "Yes" : "No") << endl;
}