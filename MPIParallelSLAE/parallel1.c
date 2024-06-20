#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define epsilon 0.00001
#define sizeN 1900

double* GetPartAMatrix(double* A, int* countElem, int* disElem, int rank, int countProcess) {

    int part = sizeN / countProcess;
    int remainder = sizeN % countProcess;

    int start, end;
    if (rank < remainder) {
        start = (part + 1) * rank;
        end = start + part + 1;
    }
    else {
        start = part * rank + remainder;
        end = start + part;
    }

    int sizePart = end - start;


    MPI_Allgather(&sizePart, 1, MPI_INTEGER, countElem, 1, MPI_INTEGER, MPI_COMM_WORLD);
    for (size_t i = 0; i < countProcess; i++)
    {
        int a = 0;
        for (int j = 0; j < i; j++) {
            a += countElem[j];
        }
        disElem[i] = a;
    }

    double* partA = (double*)malloc(sizeof(double) * sizePart*sizeN);
    for (int i = 0; i < sizePart; i++) {
        for (int j = 0; j < sizeN; j++) {
            partA[i * sizeN + j] = A[(disElem[rank]+i) * sizeN + j];
        }
    }
    return partA;
}

double* MulMatAndVec(double* partA, double* x, int size) {
    double* res = (double*)malloc(sizeof(double)*size);
    double cell = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < sizeN; j++) {
            cell += partA[i * sizeN + j] * x[j];
        }
        res[i] = cell;
        cell = 0;
    }
    return res;
}

double* DivVectors(double* vectorA, double* vectorB, int size) {
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        res[i] = vectorA[i] - vectorB[i];
    }
    return res;
}

double ScalarMul(double* vectorA, double* vectorB, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += vectorA[i] * vectorB[i];
    }
    return res;
}

double* MulScalarAndVec(double* vector, double scalar, int size) {
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++)
    {
        res[i] = vector[i] * scalar;
    }
    return res;
}

double ModuleVector(double* vector, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += vector[i] * vector[i];
    }
    return sqrt(res);
}


void Calc(double* partA, double* x, double* b, int rank, int countProcess, int* countElem, int* disElem) {
    int stop = 1;
    int sizeSegment = countElem[rank];
    int dis = disElem[rank];

    while (stop) {
        double* partY = MulMatAndVec(partA, x, sizeSegment);
        partY = DivVectors(partY, &b[dis], sizeSegment);

        double* Y = (double*)malloc(sizeof(double) * sizeN);
        MPI_Allgatherv(partY, sizeSegment, MPI_DOUBLE, Y, countElem, disElem, MPI_DOUBLE, MPI_COMM_WORLD);

        double* partAy = MulMatAndVec(partA, Y, sizeSegment);

        double* Ay = (double*)malloc(sizeof(double) * sizeN);
        MPI_Allgatherv(partAy, sizeSegment, MPI_DOUBLE, Ay, countElem, disElem, MPI_DOUBLE, MPI_COMM_WORLD);

        double tau = 0;
        if (rank == 0) {
            double numinator = ScalarMul(Y, Ay, sizeN);
            double denominator = ScalarMul(Ay, Ay, sizeN);

            tau = numinator / denominator;

            Y = MulScalarAndVec(Y, tau, sizeN);
            x = DivVectors(x, Y, sizeN);
        }
        MPI_Bcast(x, sizeN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double* partNumVector = MulMatAndVec(partA, x, sizeSegment);
        partNumVector = DivVectors(partNumVector, &b[dis], sizeSegment);
        double partNum = ScalarMul(partNumVector, partNumVector, sizeSegment);
        partNum = sqrt(partNum);
        double numinator = 0;
        double denominator = 0;
        if (rank == 0) {
            denominator = ModuleVector(b, sizeN);
        }
        MPI_Reduce(&partNum, &numinator, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (numinator / denominator < epsilon) {
                stop = 0;
            }
        }
        MPI_Bcast(&stop, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

int main() {

	double* A = (double*)malloc(sizeof(double) * sizeN * sizeN);
	double* x = (double*)malloc(sizeof(double) * sizeN);
	double* b = (double*)malloc(sizeof(double) * sizeN);
	double* u = (double*)malloc(sizeof(double) * sizeN);

    for (int i = 0; i < sizeN; i++) {
        x[i] = 0;
        u[i] = sin(2 * 3.14 * i / sizeN);
        for (int j = 0; j < sizeN; j++) {
            A[i * sizeN + j] = 1;
            if (i == j) A[i * sizeN + j] = 2;
        }
    }

    for (int i = 0; i < sizeN; i++) {
        double cell = 0;
        for (int j = 0; j < sizeN; j++) {
            cell += A[i * sizeN + j] * u[j];
        }
        b[i] = cell;
    }

    MPI_Init(NULL, NULL);


    int rank, countProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &countProcess);

    int* countElemInProcess = (int*)malloc(sizeof(int) * countProcess);
    int* disElem = (int*)malloc(sizeof(int) * countProcess);

    double* partA = GetPartAMatrix(A, countElemInProcess, disElem, rank, countProcess);

    double start = MPI_Wtime();
    
    Calc(partA, x, b, rank, countProcess, countElemInProcess, disElem);

    double end = MPI_Wtime();
    double time = end - start;
    double maxTime;
    MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%f", maxTime);

    MPI_Finalize();

}