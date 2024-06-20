#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define L 100
#define LISTS_COUNT 25
#define TASK_COUNT 2000

pthread_mutex_t mutex;
pthread_t threads[2];

int* tasks;

int procCount;
int procRank;
int REMAINING_TASKS;

double SUMMARY_DISBALANCE = 0;

void initTasks(int* tasks, int taskCount, int iter) {
    for (int i = 0; i < taskCount; i++) {
        tasks[i] = abs(procRank - (iter % procCount)) * L;
    }
}

void doTasks(int* tasks) {
    for (int i = 0; i < REMAINING_TASKS; i++) {
        pthread_mutex_lock(&mutex);
        int weight = tasks[i];
        pthread_mutex_unlock(&mutex);
        double res = 0;
        for (int j = 0; j < weight; j++) {
            res += sqrt(j);
        }
    }
    pthread_mutex_lock(&mutex);
    REMAINING_TASKS = 0;
    pthread_mutex_unlock(&mutex);
}

void* Solver(void* args) {
    tasks = (int*)malloc(sizeof(int) * TASK_COUNT);
    double startTime, finishTime, iterationDuration, shortest, longest;
    for (int i = 0; i < LISTS_COUNT; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        REMAINING_TASKS = TASK_COUNT;
        startTime = MPI_Wtime();
        initTasks(tasks, TASK_COUNT, i);
        doTasks(tasks);

        int response;
        for (int procIdx = 0; procIdx < procCount; procIdx++) {
            if (procIdx != procRank) {
                MPI_Send(&procRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);
                MPI_Recv(&response, 1, MPI_INT, procIdx, SENDING_TASK_COUNT,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (response != NO_TASKS_TO_SHARE) {
                    MPI_Recv(tasks, response, MPI_INT, procIdx, SENDING_TASKS,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pthread_mutex_lock(&mutex);
                    REMAINING_TASKS = response;
                    pthread_mutex_unlock(&mutex);
                    doTasks(tasks);
                }
            }
        }
        finishTime = MPI_Wtime();
        iterationDuration = finishTime - startTime;
        MPI_Allreduce(&iterationDuration, &longest, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&iterationDuration, &shortest, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);

        printf("%s %d iteration do %f %f in time %f %s\n", ANSI_GREEN, i,
               longest, shortest, iterationDuration, ANSI_RESET);
        SUMMARY_DISBALANCE += (longest - shortest) / longest;
    }
    int Signal = SOLVER_FINISHED_WORK;
    if (procCount != 1)
        MPI_Send(&Signal, 1, MPI_INT, procRank, 888, MPI_COMM_WORLD);
}

void* Reciever(void* args) {
    int askingProcRank, answer, request;
    MPI_Status status;
    while (1) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD,
                 &status);

        if (request == SOLVER_FINISHED_WORK) {
            return NULL;
        }
        askingProcRank = request;
        pthread_mutex_lock(&mutex);
        if (REMAINING_TASKS > TASK_COUNT / procCount) {
            int old = REMAINING_TASKS;
            answer = REMAINING_TASKS / procCount;
            REMAINING_TASKS -= answer;

            printf("%s sharing %d from %d to %d %s\n", ANSI_CYAN, answer,
                   procRank, askingProcRank, ANSI_RESET);

            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT,
                     MPI_COMM_WORLD);
            MPI_Send(&tasks[REMAINING_TASKS], answer, MPI_INT, askingProcRank,
                     SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT,
                     MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);
    }
}

int main() {
    int thr;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thr);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    pthread_mutex_init(&mutex, NULL);
    double start = MPI_Wtime();
    pthread_create(&threads[0], NULL, Solver, NULL);
    if (procCount != 1) {
        pthread_create(&threads[1], NULL, Reciever, NULL);
        pthread_join(threads[1], NULL);
    }

    pthread_join(threads[0], NULL);

    double finish = MPI_Wtime() - start;
    double time;
    MPI_Reduce(&finish, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (procRank == 0) {
        printf("\n---------------\nTime %f\n", time);
        printf("%s Disbalance %f%% %s\n", ANSI_RED,
               SUMMARY_DISBALANCE / LISTS_COUNT * 100, ANSI_RESET);
    }

    MPI_Finalize();
}
