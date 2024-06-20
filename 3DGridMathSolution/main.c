#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//$$N = N_x = N_y = N_z$$
#define N 512

#define a 10e5
#define epsilon 10e-8
#define idx(i, j, k) N* N*(i) + N*(j) + k

#define D_X 2
#define D_Y 2
#define D_Z 2

#define X_0 -1
#define Y_0 -1
#define Z_0 -1

double H_X = D_X / (double)(N - 1);
double H_Y = D_Y / (double)(N - 1);
double H_Z = D_Z / (double)(N - 1);

double H_X2 = D_X / (double)(N - 1) * D_X / (double)(N - 1);
double H_Y2 = D_Y / (double)(N - 1) * D_Y / (double)(N - 1);
double H_Z2 = D_Z / (double)(N - 1) * D_Z / (double)(N - 1);

int procRank = 0;
int procNum = 0;

//$$\phi(x,y,z)= x^2+y^2+z^2$$
double phi(double x, double y, double z) { return x * x + y * y + z * z; }

//$$\rho(x,y,z)=6-a*\phi(x,y,z)$$
double ro(double x, double y, double z) { return 6 - a * phi(x, y, z); }

//$$x_i=x_o+i*h_x$$
double X(int i) { return (X_0 + i * H_X); }

//$$y_i=y_o+i*h_y$$
double Y(int i) { return (Y_0 + i * H_Y); }

//$$z_i=z_o+i*h_z$$
double Z(int i) { return (Z_0 + i * H_Z); }

//$$\Delta=max_{i,j,k}|\phi^m - \phi^* |$$
double comparePhi(double* calcLayer, int layerHeight) {
    double delta = DBL_MIN;
    double x, y, z;
    for (int i = 1; i < layerHeight - 1; i++) {
        x = X(i + procRank * layerHeight);
        for (int j = 1; j < N - 1; j++) {
            y = Y(j);
            for (int k = 1; k < N - 1; k++) {
                z = Z(k);
                delta =
                    fmax(delta, fabs(calcLayer[idx(i, j, k)] - phi(x, y, z)));
            }
        }
    }
    return delta;
}

//$$\phi|_\Omega = x^2+y^2+z^2,  \phi^0=0$$
void initializePhi(int LayerHeight, double* currentLayer) {
    for (int i = 0; i < LayerHeight + 2; i++) {
        int layerZCoord = i + procRank * LayerHeight - 1;
        double z = Z(layerZCoord);
        for (int j = 0; j < N; j++) {
            double y = Y(j);
            for (int k = 0; k < N; k++) {
                double x = Z(k);
                if (j == 0 || j == N - 1 || k == 0 || k == N - 1 || z == Z_0 ||
                    z == Z_0 + D_Z) {
                    currentLayer[idx(i, j, k)] = phi(x, y, z);
                }
            }
        }
    }
}

void printfLayer(int LayerHeight, double* currentLayer) {
    for (int i = 0; i < LayerHeight + 2; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%f ", currentLayer[idx(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
}

double UpdateLayer(int startZLayer, int layerIdx, double* currentLayer,
                   double* currentLayerBuf) {
    int absolutZCoord = startZLayer + layerIdx;
    double x, y, z;
    double deltaMax = DBL_MIN;

    if (absolutZCoord == 0 || absolutZCoord == N - 1) {
        memcpy(currentLayerBuf + layerIdx * N * N,
               currentLayer + layerIdx * N * N, N * N * sizeof(double));
        deltaMax = 0;
    } else {
        z = Z(absolutZCoord);
        for (int i = 0; i < N; i++) {
            y = Y(i);
            for (int j = 0; j < N; j++) {
                x = X(j);
                if (i != 0 && i != N - 1 && j != 0 && j != N - 1) {
                    currentLayerBuf[idx(layerIdx, i, j)] =
                        ((currentLayer[idx(layerIdx + 1, i, j)] +
                          currentLayer[idx(layerIdx - 1, i, j)]) /
                             H_Z2 +
                         (currentLayer[idx(layerIdx, i + 1, j)] +
                          currentLayer[idx(layerIdx, i - 1, j)]) /
                             H_X2 +
                         (currentLayer[idx(layerIdx, i, j + 1)] +
                          currentLayer[idx(layerIdx, i, j - 1)]) /
                             H_Y2 -
                         ro(x, y, z)) /
                        (2 / H_X2 + 2 / H_Y2 + 2 / H_Z2 + a);
                    deltaMax = fmax(deltaMax,
                                    fabs(currentLayerBuf[idx(layerIdx, i, j)] -
                                         currentLayer[idx(layerIdx, i, j)]));
                } else {
                    currentLayerBuf[idx(layerIdx, i, j)] =
                        currentLayer[idx(layerIdx, i, j)];
                }
            }
        }
    }
    return deltaMax;
}

int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    int part = N / procNum;
    int remainder = N % procNum;
    int start, end;
    if (procRank < remainder) {
        start = (part + 1) * procRank;
        end = start + part + 1;
    } else {
        start = part * procRank + remainder;
        end = start + part;
    }

    int layerHeight = end - start;
    int layerZCoord = procRank * layerHeight - 1;

    int layerSize = (layerHeight + 2) * N * N;
    double* currentLayer = (double*)calloc(layerSize, sizeof(double));
    double* currentLayerBuf = (double*)calloc(layerSize, sizeof(double));
    initializePhi(layerHeight, currentLayer);
    memcpy(currentLayerBuf + N * N, currentLayer + N * N,
           sizeof(double) * layerHeight * N * N);

    double globalDelta = DBL_MAX;

    MPI_Request req[4];

    double startTime = MPI_Wtime();
    do {
        double procMaxDelta = DBL_MIN;
        double tmpMaxDelta;

        tmpMaxDelta =
            UpdateLayer(layerZCoord, 1, currentLayer, currentLayerBuf);
        procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);
        tmpMaxDelta = UpdateLayer(layerZCoord, layerHeight, currentLayer,
                                  currentLayerBuf);
        procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);

        if (procRank != 0) {
            MPI_Isend(currentLayerBuf + N * N, N * N, MPI_DOUBLE, procRank - 1,
                      0, MPI_COMM_WORLD, &req[0]);

            MPI_Irecv(currentLayerBuf, N * N, MPI_DOUBLE, procRank - 1, 0,
                      MPI_COMM_WORLD, &req[1]);
        }

        if (procRank != procNum - 1) {
            MPI_Isend(currentLayerBuf + N * N * layerHeight, N * N, MPI_DOUBLE,
                      procRank + 1, 0, MPI_COMM_WORLD, &req[2]);

            MPI_Irecv(currentLayerBuf + N * N * (layerHeight + 1), N * N,
                      MPI_DOUBLE, procRank + 1, 0, MPI_COMM_WORLD, &req[3]);
        }

        for (int layerIdx = 2; layerIdx < layerHeight; layerIdx++) {
            tmpMaxDelta = UpdateLayer(layerZCoord, layerIdx, currentLayer,
                                      currentLayerBuf);
            procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);
        }

        if (procRank != procNum - 1) {
            MPI_Status st;
            MPI_Waitall(2, &req[2], &st);
        }

        if (procRank != 0) {
            MPI_Status st;
            MPI_Waitall(2, &req[0], &st);
        }

        memcpy(currentLayer, currentLayerBuf, layerSize * sizeof(double));
        MPI_Allreduce(&procMaxDelta, &globalDelta, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);

    } while (globalDelta > epsilon);
    double endTime = MPI_Wtime();

    free(currentLayerBuf);
    double procTime = endTime - startTime;
    double maxTime;
    MPI_Reduce(&procTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double delta = comparePhi(currentLayer + N * N, layerHeight);
    double maxDelta;
    MPI_Reduce(&delta, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (procRank == 0) {
        printf("Time = %f sec delta = %f\n", maxTime, maxDelta);
    }

    free(currentLayer);

    MPI_Finalize();
}