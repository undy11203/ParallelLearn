#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>    // for clock_t, clock()

#define epsilon 0.00001
#define sizeN 10

double* MulMatAndVec(double* Matrix, double* vector) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    double cell = 0;
    for (int i = 0; i < sizeN; i++)
    {
        for (int j = 0; j < sizeN; j++)
        {
            cell += Matrix[i * sizeN + j] * vector[j];
        }
        res[i] = cell;
        cell = 0;
    }
    return res;
}

double* DivVectors(double* vectorA, double* vectorB) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    for (int i = 0; i < sizeN; i++)
    {
        res[i] = vectorA[i] - vectorB[i];
    }
    return res;
}

double ScalarMul(double* vectorA, double* vectorB) {
    double res = 0;
    for (int i = 0; i < sizeN; i++)
    {
        res += vectorA[i] * vectorB[i];
    }
    return res;
}

double* MulScalarAndVec(double* vector, double scalar) {
    double* res = (double*)malloc(sizeof(double) * sizeN);
    for (int i = 0; i < sizeN; i++)
    {
        res[i] = vector[i] * scalar;
    }
    return res;
}

double ModuleVector(double* vector) {
    double res = 0;
    for (int i = 0; i < sizeN; i++)
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

void Calc(double* A, double* x, double* b){
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
    for(int i = 0; i <sizeN; i++){
        printf("%f\n", x[i]);
    }
}

int main()
{

    double* A = (double*)malloc(sizeof(double) * sizeN * sizeN);
    double* b = (double*)malloc(sizeof(double) * sizeN);
    double* x = (double*)malloc(sizeof(double) * sizeN);
    double* u = (double*)malloc(sizeof(double) * sizeN);

    for (int i = 0; i < sizeN; i++) {
        x[i] = 0;
        u[i] = sin(2 * 3.14 * i / sizeN);
        for (int j = 0; j < sizeN; j++) {
            A[i * sizeN + j] = i;
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

    clock_t start = clock();

    Calc(A, x, b);

    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("%f", time);
}
 
