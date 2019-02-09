#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
#include<stdio.h>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <stack>
#include<sstream>
#include<math.h>
#include<stdlib.h>
#include <cusolverDn.h>
#include<cassert>
#include<random>
#include<math.h>
#include <chrono>
#include <random>
#include"QR.cu"
//#include"preprocessing.cu"
using namespace std;
//void preprocessing(float * , int ,int );
__global__ void ker_preprocessing(float*,float*,int ,int,int,int);
void compute_q(float *, float* ,int ,int ,int,int,char);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

float * normal_random_generator_sigma(int n,int l)
{
    float* sigma=new float[n*l];
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);

    for(int i=0;i<l;i++)
    {
       std::normal_distribution<double> distribution (0.0,1.0);
       for (int j=0;j<n;j++)
          {
              sigma[i*n+j]=distribution(generator);
          }
    }

    return sigma;
}//end of normal_random_generator_sigma




int CorMat_2_decomposition(float * BOLD, int N, int L,int windowsize,int windowstep,int LL,char pr)//( float * BOLD, int N, int L,int Nrows,int Ncols,int LL)
{

    int slide_num=1;
    if(windowstep!=0)
         slide_num = (L-windowsize)/windowstep +1;
    

     long long M1 = (N-1); //computing the  size of correlaion matrix
     M1 *= N;
     M1 /= 2;

     cudaError_t cudaStat;
     cublasStatus_t stat;
     long long total=N*N;//size of total correlation matrix

     float * total_cormat = new float [total];

     float * devBOLD; //Allocating space in GPU for storing fMRI data
     cudaStat = cudaMalloc ((void**)&devBOLD, sizeof(float) * L * N) ;

     if (cudaStat != cudaSuccess)
     {
        cout<<"Error in Cuda Malloc";
        return cudaStat;
     }
     float* dev_windowedbold;
     cudaStat = cudaMalloc ((void**)&dev_windowedbold, sizeof(float) * windowsize * N) ;

     if (cudaStat != cudaSuccess)
     {
        cout<<"Error in Cuda Malloc";
        return cudaStat;
     }
     cudaMemcpy( devBOLD,BOLD, sizeof(float) *N*L, cudaMemcpyHostToDevice);
    int stepn=0;
//float* address;

    float* devCormat;//allocating space in GPU for whole correlation matrix
    cudaMalloc ( (void**)&devCormat, sizeof(float) * total) ;

    cublasHandle_t handle;
    stat = cublasCreate(&handle) ;
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error in creating cublas handle";
        return stat;
    }


    const float alpha = 1.0;
    const float beta = 0.0;

for(int wstep=0;wstep<slide_num;wstep++)
{


    ker_preprocessing<<<N,32,(windowsize+1)*sizeof(float)>>>(devBOLD,dev_windowedbold,N,L, windowsize, stepn);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();
    stepn+=windowstep ;

    stat=  cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, N,N,windowsize,  &alpha, dev_windowedbold, windowsize, dev_windowedbold, windowsize, &beta, devCormat, N);
    cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error performing multiplication";
        return stat;
    }

    float* my_sigma=normal_random_generator_sigma(N,L);
    float* devSigma;//allocating space in GPU for whole correlation matrix
    cudaMalloc ( (void**)&devSigma, sizeof(float) *N*L);
    cudaMemcpy (devSigma,my_sigma, sizeof(float) *N*L, cudaMemcpyHostToDevice );

    compute_q(devSigma, devCormat,N,L,LL,wstep,pr);

    cudaFree (devSigma);
    delete []my_sigma;
}
return 0;
}
