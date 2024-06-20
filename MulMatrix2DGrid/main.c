#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "Func.h"

// height x width
// A[M x N]
// B[N x K]
// C[M x K]

int main(int argc, char* argv[]) {
    int SizeVerM = 2000;
    int SizeHorN = 3000;
    int SizeVerN = 3000;
    int SizeHorK = 2500;
    double *A, *B, *C, *pA, *pTransB, *pC;

    MPI_Init(NULL, NULL);

    int rank, countProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &countProcess);

    double start = MPI_Wtime();

    if (rank == 0) {
        InitMatrix(&A, &B, &C, SizeVerM, SizeHorN, SizeVerN, SizeHorK);
    }

    MPI_Comm grid;
    int width = strtol(argv[1], NULL, 10);
    int height = strtol(argv[2], NULL, 10);

    Create2DTopological(width, height, &grid);

    int sizeVerPartA, sizeHorPartA, sizeVerPartB, sizeHorPartB;
    pA = GetPartMatrixA(A, grid, height, width, SizeVerM, SizeHorN,
                        &sizeVerPartA, &sizeHorPartA);

    pTransB = GetPartMatrixB(B, grid, height, width, SizeVerN, SizeHorK,
                             &sizeVerPartB, &sizeHorPartB);

    GetMul(pA, pTransB, &pC, sizeVerPartA, sizeHorPartA, sizeVerPartB,
           sizeHorPartB);

    GatherSubArray(C, pC, sizeVerPartA, sizeVerPartB, SizeVerM, SizeHorK,
                   height, width, grid);

    double end = MPI_Wtime();
    double maxTime = end - start;
    if (rank == 0) printf("%f\n", maxTime);

    MPI_Finalize();
}
