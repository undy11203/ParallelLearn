#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>    

#define epsilon 0.00001
#define sizeN 1900

double* MulMatAndVec(double* Matrix, double* vector) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < sizeN; i++)
    {
        for (j = 0; j < sizeN; j++)
        {
            res[i] += Matrix[i * sizeN + j] * vector[j];
        }
    }
    return res;
}

double* DivVectors(double* vectorA, double* vectorB) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    int i;
    #pragma omp parallel for
    for (i = 0; i < sizeN; i++)
    {
        res[i] = vectorA[i] - vectorB[i];
    }
    return res;
}

double ScalarMul(double* vectorA, double* vectorB) {
    double res = 0;
    int i;
    #pragma omp parallel for reduction (+:res)
    for (i = 0; i < sizeN; i++)
    {
        res += vectorA[i] * vectorB[i];
    }
    return res;
}

double* MulScalarAndVec(double* vector, double scalar) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    int i;
    #pragma omp parallel for
    for (i = 0; i < sizeN; i++)
    {
        res[i] = vector[i] * scalar;
    }
    return res;
}

double ModuleVector(double* vector) {
    double res = 0;
    int i;
    #pragma omp parallel for reduction (+:res)
    for (i = 0; i < sizeN; i++)
    {
        res += vector[i] * vector[i];
    }
    return sqrt(res);
}

char IsStop(double* vectorA, double* vectorB) {
    double* numinatorVec = DivVectors(vectorA, vectorB);
    double numinator = ModuleVector(numinatorVec);
    double denominator = ModuleVector(vectorB);

    if (numinator / denominator < epsilon) {
        return 1;
    }

    return 0;
}

double* Calc(double* A, double* x, double* b){
    char isStopValue = 0;
    while (!isStopValue)
    {
        double* Y = MulMatAndVec(A, x);
        Y = DivVectors(Y, b);

        double* Ay = MulMatAndVec(A, Y);

        double numinator = ScalarMul(Y, Ay);
        double denominator = ScalarMul(Ay, Ay);
        double tau = numinator / denominator;

        Y = MulScalarAndVec(Y, tau);
        x = DivVectors(x, Y);

        double* vec1 = MulMatAndVec(A, x);
        isStopValue = IsStop(vec1, b);

        if (isStopValue) {
            break;
        }
    }

    return x;
}

int main()
{

    double* A = (double*)malloc(sizeof(double) * sizeN * sizeN);
    double* b = (double*)malloc(sizeof(double) * sizeN);
    double* x = (double*)malloc(sizeof(double) * sizeN);
    double* u = (double*)malloc(sizeof(double) * sizeN);

    for (int i = 0; i < sizeN; i++)
    {
        x[i] = 0;
        u[i] = sin(2 * 3.14 * i / sizeN);
        for (int j = 0; j < sizeN; j++) {
            A[i * sizeN + j] = 1;
            if (i == j) A[i * sizeN + j] = 2;
        }
    }
    for (int i = 0; i < sizeN; i++)
    {
        double cell = 0;
        for (int j = 0; j < sizeN; j++) {
            cell += A[i * sizeN + j] * u[j];
        }
        b[i] = cell;
    }

    double start = omp_get_wtime();

    x = Calc(A, x, b);

    double end = omp_get_wtime();
    double time = end - start;

    printf("%f", time);

    FILE* fp = fopen("par.txt", "w");
    for(int i = 0; i < sizeN; i++){
        fprintf(fp, "%f\n", x[i]);
    }
}

