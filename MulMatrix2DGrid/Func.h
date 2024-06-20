#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void InitMatrix(double** A, double** B, double** C, int SizeVerM, int SizeHorN,
                int SizeVerN, int SizeHorK) {
    *A = (double*)malloc(sizeof(double) * SizeVerM * SizeHorN);
    *B = (double*)malloc(sizeof(double) * SizeVerN * SizeHorK);
    *C = (double*)malloc(sizeof(double) * SizeVerM * SizeHorK);

    for (size_t i = 0; i < SizeVerM; i++) {
        for (size_t j = 0; j < SizeHorN; j++) {
            (*A)[i * SizeHorN + j] = j;
        }
    }

    for (size_t i = 0; i < SizeVerN; i++) {
        for (size_t j = 0; j < SizeHorK; j++) {
            (*B)[i * SizeHorK + j] = j;
        }
    }

    for (size_t i = 0; i < SizeVerM; i++) {
        for (size_t j = 0; j < SizeHorK; j++) {
            (*C)[i * SizeHorK + j] = 10;
        }
    }
}

void Create2DTopological(int width, int height, MPI_Comm* grid) {
    int dims[2] = {width, height};
    int period[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, grid);
}

double* GetPartMatrixA(double* Mat, MPI_Comm grid, int heightProccess,
                       int widthProcess, int SizeVerM, int SizeHorN,
                       int* sizeVer, int* sizeHor) {
    int rank, gridCoords[2];  //(x, y)
    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 2, gridCoords);

    MPI_Comm col;
    int dims[2] = {0, 1};
    MPI_Cart_sub(grid, dims, &col);

    double* pMat;
    int *countElem = (int*)malloc(sizeof(int) * heightProccess), *disElem;
    if (gridCoords[0] == 0) {
        int part = SizeVerM / heightProccess;
        int remainder = SizeVerM % heightProccess;

        int colRank;
        MPI_Comm_rank(col, &colRank);

        int start, end;
        if (gridCoords[1] < remainder) {
            start = (part + 1) * gridCoords[1];
            end = start + part + 1;
        } else {
            start = part * gridCoords[1] + remainder;
            end = start + part;
        }

        int sizePart = (end - start) * SizeHorN;
        int* disElem = (int*)malloc(sizeof(int) * heightProccess);
        MPI_Allgather(&sizePart, 1, MPI_INTEGER, countElem, 1, MPI_INTEGER,
                      col);
        for (size_t i = 0; i < heightProccess; i++) {
            int a = 0;
            for (int j = 0; j < i; j++) {
                a += countElem[j];
            }
            disElem[i] = a;
        }

        pMat = (double*)malloc(sizeof(double) * sizePart);
        MPI_Scatterv(Mat, countElem, disElem, MPI_DOUBLE, pMat,
                     countElem[colRank], MPI_DOUBLE, 0, col);
    }

    MPI_Bcast(countElem, heightProccess, MPI_INT, 0, grid);

    if (gridCoords[0] != 0) {
        pMat = (double*)malloc(sizeof(double) * countElem[gridCoords[1]]);
    }

    MPI_Comm row;
    dims[0] = 1;
    dims[1] = 0;

    MPI_Cart_sub(grid, dims, &row);

    int idx = gridCoords[1];
    MPI_Bcast(pMat, countElem[idx], MPI_DOUBLE, 0, row);
    *sizeVer = countElem[idx] / SizeHorN;
    *sizeHor = SizeHorN;

    return pMat;
}

double* GetPartMatrixB(double* Mat, MPI_Comm grid, int heightProccess,
                       int widthProcess, int SizeVerN, int SizeHorK,
                       int* sizeVer, int* sizeHor) {
    int rank, gridCoords[2];  //(x, y)
    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 2, gridCoords);

    MPI_Comm row;
    int dims[2] = {1, 0};
    MPI_Cart_sub(grid, dims, &row);

    double* pMat;
    int *countElem = (int*)malloc(sizeof(int) * widthProcess), *disElem;
    if (gridCoords[1] == 0) {
        int part = SizeHorK / widthProcess;
        int remainder = SizeHorK % widthProcess;

        int rowRank;
        MPI_Comm_rank(row, &rowRank);

        int start;
        int end;
        if (gridCoords[0] < remainder) {
            start = (part + 1) * gridCoords[0];
            end = start + part + 1;
        } else {
            start = part * gridCoords[0] + remainder;
            end = start + part;
        }
        int sizePart = end - start;

        MPI_Datatype column, colType;
        MPI_Type_vector(SizeVerN, 1, SizeHorK, MPI_DOUBLE, &column);
        MPI_Type_commit(&column);
        MPI_Type_create_resized(column, 0, 1 * sizeof(double), &colType);
        MPI_Type_commit(&colType);

        int* disElem = (int*)malloc(sizeof(int) * widthProcess);
        MPI_Allgather(&sizePart, 1, MPI_INT, countElem, 1, MPI_INT, row);

        for (size_t i = 0; i < widthProcess; i++) {
            int a = 0;
            for (int j = 0; j < i; j++) {
                a += countElem[j];
            }
            disElem[i] = a;
        }

        pMat = (double*)malloc(sizeof(double) * sizePart * SizeVerN);

        MPI_Scatterv(Mat, countElem, disElem, colType, pMat,
                     countElem[rowRank] * SizeVerN, MPI_DOUBLE, 0, row);

        MPI_Type_free(&column);
        MPI_Type_free(&colType);
    }

    MPI_Bcast(countElem, widthProcess, MPI_INT, 0, grid);

    if (gridCoords[1] != 0) {
        pMat = (double*)malloc(sizeof(double) * countElem[gridCoords[0]] *
                               SizeVerN);
    }
    MPI_Comm col;
    dims[0] = 0;
    dims[1] = 1;

    MPI_Cart_sub(grid, dims, &col);

    MPI_Bcast(pMat, countElem[gridCoords[0]] * SizeVerN, MPI_DOUBLE, 0, col);

    *sizeVer = countElem[gridCoords[0]];
    *sizeHor = SizeVerN;

    return pMat;
}

void GetMul(double* pA, double* pTransB, double** pC, int sizeVerPartA,
            int sizeHorPartA, int sizeVerPartB, int sizeHorPartB) {
    *pC = (double*)malloc(sizeof(double) * sizeVerPartA * sizeVerPartB);
    for (int j = 0; j < sizeVerPartB; j++) {
        for (int k = 0; k < sizeVerPartA; k++) {
            double result = 0;
            for (int i = 0; i < sizeHorPartA; i++) {
                result +=
                    pA[k * sizeHorPartA + i] * pTransB[j * sizeHorPartB + i];
            }
            (*pC)[k * sizeVerPartB + j] = result;
        }
    }
}

void GatherSubArray(double* C, double* pC, int pHeight, int pWidth, int height,
                    int width, int heightProccess, int widthProccess,
                    MPI_Comm grid) {
    int size, rank, gridCoords[2];
    MPI_Comm_size(grid, &size);
    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 2, gridCoords);

    int* arrayCount = (int*)malloc(sizeof(int) * size);
    int subMatSize = pHeight * pWidth;
    MPI_Gather(&subMatSize, 1, MPI_INT, arrayCount, 1, MPI_INT, 0, grid);

    MPI_Datatype* arrayOfType =
        (MPI_Datatype*)malloc(sizeof(MPI_Datatype) * size);

    MPI_Datatype subArrayType;
    int arrSubSize[2] = {pHeight, pWidth};
    int arrStart[2];

    int part = height / heightProccess;
    int remainder = height % heightProccess;
    int start, end;
    if (gridCoords[1] < remainder) {
        start = (part + 1) * gridCoords[1];
    } else {
        start = part * gridCoords[1] + remainder;
    }
    arrStart[0] = start;

    part = width / widthProccess;
    remainder = width % widthProccess;
    if (gridCoords[0] < remainder) {
        start = (part + 1) * gridCoords[0];
    } else {
        start = part * gridCoords[0] + remainder;
    }
    arrStart[1] = start;

    int arrSize[2] = {height, width};
    int* arrSubSizeToMain = (int*)malloc(sizeof(int) * size * 2);
    int* arrStartToMain = (int*)malloc(sizeof(int) * size * 2);

    MPI_Gather(arrSubSize, 2, MPI_INT, arrSubSizeToMain, 2, MPI_INT, 0, grid);
    MPI_Gather(arrStart, 2, MPI_INT, arrStartToMain, 2, MPI_INT, 0, grid);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            MPI_Type_create_subarray(2, arrSize, &arrSubSizeToMain[i * 2],
                                     &arrStartToMain[i * 2], MPI_ORDER_C,
                                     MPI_DOUBLE, &arrayOfType[i]);
            MPI_Type_commit(&arrayOfType[i]);
        }
    }

    MPI_Request req1;
    MPI_Isend(pC, pHeight * pWidth, MPI_DOUBLE, 0, 0, grid, &req1);

    if (rank == 0) {
        MPI_Request* reqs = (MPI_Request*)malloc(size * sizeof(MPI_Request));
        MPI_Status* stats = (MPI_Status*)malloc(size * sizeof(MPI_Status));

        for (int i = 0; i < size; i++) {
            MPI_Irecv(C, arrayCount[i], arrayOfType[i], i, 0, grid, &reqs[i]);
        }

        MPI_Waitall(size, reqs, stats);

        for (int i = 0; i < size; i++) {
            MPI_Type_free(&arrayOfType[i]);
        }

        free(reqs);
        free(stats);
    }
}

