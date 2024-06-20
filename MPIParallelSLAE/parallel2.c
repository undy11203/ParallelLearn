#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define epsilon 0.00001
#define sizeN 1900

#define min(a,b) ((a<b)? (a) : (b))

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

    double* partA = (double*)malloc(sizeof(double) * sizePart * sizeN);
    for (int i = 0; i < sizePart; i++) {
        for (int j = 0; j < sizeN; j++) {
            partA[i * sizeN + j] = A[(disElem[rank] + i) * sizeN + j];
        }
    }
    return partA;
}

double* GetPartVector(double* vector, int* countElemInProcess, int* disElem, int rank, int countProcess) {
    double* res = (double*)malloc(sizeof(double) * countElemInProcess[rank]);
    for (size_t i = 0; i < countElemInProcess[rank]; i++)
    {
        res[i] = vector[disElem[rank] + i];
    }
    return res;
}

double* MulPartMatAndPartVec(double* partA, double* partX, int size) {
    double* res = (double*)calloc(sizeN, sizeof(double));

    for(int i = 0; i < size; i++){
        for(int j = 0; j < sizeN; j++){
            res[j] += partA[i*sizeN+j]*partX[i];
        }
    }
    return res;
}

double* MulMatAndVec(double* partA, double* x, int size) {
    double* res = (double*)calloc(size, sizeof(double));
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

double* DivVecAndPartVec(double* vectorA, double* vectorB, int sizeSegment, int disElem) {
    for (int i = disElem; i < sizeSegment+disElem; i++) {
        vectorA[i] -= vectorB[i-disElem];
    }
    return vectorA;
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

void Calc(double* partA, double* partX, double* partB, int rank, int countProcess, int* countElem, int* disElem) {
    int stop = 1;
    int sizeSegment = countElem[rank];
    int dis = disElem[rank];
    int count = 0;

    while (stop) {
        count++;

        double* partY = MulPartMatAndPartVec(partA, partX, sizeSegment);
        partY = DivVecAndPartVec(partY, partB, sizeSegment, disElem[rank]);

        double* Y = (double*)malloc(sizeof(double) * sizeN);
        MPI_Allreduce(partY, Y, sizeN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double* partAy = MulPartMatAndPartVec(partA, &Y[dis], sizeSegment);

        double* Ay = (double*)malloc(sizeof(double) * sizeN);
        MPI_Allreduce(partAy, Ay, sizeN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tau = 0;
        if (rank == 0) {
            double numinator = ScalarMul(Y, Ay, sizeN);
            double denominator = ScalarMul(Ay, Ay, sizeN);

            tau = numinator / denominator;
            Y = MulScalarAndVec(Y, tau, sizeN);
        }

        MPI_Bcast(Y, sizeN, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        partX = DivVectors(partX, &Y[disElem[rank]], sizeSegment);

        double* partNumVector = MulPartMatAndPartVec(partA, partX, sizeSegment); //fix

        partNumVector = DivVecAndPartVec(partNumVector, partB, sizeSegment, disElem[rank]);
        double* numVec = (double*)malloc(sizeof(double) * sizeN);
        
        MPI_Allreduce(partNumVector, numVec, sizeN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double numinator = ScalarMul(numVec, numVec, sizeN);
        numinator = sqrt(numinator);

        double partDenom = ScalarMul(partB, partB, sizeSegment);
        double denominator = 0;
        MPI_Reduce(&partDenom, &denominator, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        denominator = sqrt(denominator);
        
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

    for(int i = 0; i<sizeN; i++){
        for(int j = i + 1; j<sizeN; j++){
            double temp = A[i*sizeN+j];
            A[i*sizeN+j] = A[j*sizeN+i];
            A[j*sizeN+i] = temp;
        }
    }

    MPI_Init(NULL, NULL);

    int rank, countProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &countProcess);

    int* countElemInProcess = (int*)malloc(sizeof(int) * countProcess);
    int* disElem = (int*)malloc(sizeof(int) * countProcess);

    double* partA = GetPartAMatrix(A, countElemInProcess, disElem, rank, countProcess);
    double* partX = GetPartVector(x, countElemInProcess, disElem, rank, countProcess);
    double* partB = GetPartVector(b, countElemInProcess, disElem, rank, countProcess);

    double start = MPI_Wtime();

    Calc(partA, partX, partB, rank, countProcess, countElemInProcess, disElem);

    double end = MPI_Wtime();
    double maxTime = end - start;
    if(rank == 0) printf("%f\n", maxTime);

    MPI_Finalize();

}
