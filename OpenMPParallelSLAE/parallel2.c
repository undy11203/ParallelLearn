#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>    

#define epsilon 0.00001
#define sizeN 1900

double* Calc(double* A, double* x, double* b){
    double* Y = (double*)malloc(sizeof(double)*sizeN);
    double* Ay = (double*)malloc(sizeof(double)*sizeN);
    double* vec1 = (double*)malloc(sizeof(double)*sizeN);
    double numinator;
    double denominator;
    double tau;

    char isStopValue;
    int i,j;

    #pragma omp parallel private(i,j)
    {
        while (!isStopValue)
        {
            #pragma omp for
            for(i = 0; i <sizeN; i++){
                Y[i] = 0;
                vec1[i] = 0;
                Ay[i] = 0;
            }
            
            #pragma omp for
            for (i = 0; i < sizeN; i++)
            {
                for (j = 0; j < sizeN; j++)
                {
                    Y[i] += A[i * sizeN + j] * x[j];
                }
            }

            #pragma omp for 
            for (i = 0; i < sizeN; i++)
            {
                Y[i] = Y[i] - b[i];
            }

    
            #pragma omp for
            for (i = 0; i < sizeN; i++)
            {
                for (j = 0; j < sizeN; j++)
                {
                    Ay[i] += A[i * sizeN + j] * Y[j];
                }
            }

            numinator = 0;
            #pragma omp parallel for reduction (+:numinator)
            for (i = 0; i < sizeN; i++)
            {
                numinator += Ay[i] * Y[i];
            }
    
            denominator = 0;
            #pragma omp parallel for reduction (+:denominator)
            for (i = 0; i < sizeN; i++)
            {
                denominator += Ay[i] * Ay[i];
            }

            tau = numinator / denominator;

            #pragma omp for
            for (i = 0; i < sizeN; i++)
            {
                x[i] -= tau * Y[i];
            }

            #pragma omp for
            for (i = 0; i < sizeN; i++)
            {
                for (j = 0; j < sizeN; j++)
                {
                    vec1[i] += A[i * sizeN + j] * x[j];
                }
            }

            #pragma omp for
            for (i = 0; i < sizeN; i++)
            {
                vec1[i] = vec1[i] - b[i];
            }

            numinator = 0;
            #pragma omp parallel for reduction (+:numinator)
            for (int i = 0; i < sizeN; i++)
            {
                numinator += vec1[i] * vec1[i];
            }
            #pragma omp atomic write
            numinator = sqrt(numinator);
            denominator = 0;
            #pragma omp parallel for reduction (+:denominator)
            for (int i = 0; i < sizeN; i++)
            {
                denominator += b[i] * b[i];
            }

            #pragma omp atomic write
            denominator = sqrt(denominator);
            #pragma omp critical
            if(numinator/denominator < epsilon){
                isStopValue = 1;
            }
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


