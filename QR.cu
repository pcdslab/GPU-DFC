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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iomanip>
#include <assert.h>


using namespace std;

void diff(float* original,float* recon,int m, int n)
{
    long long total=m;
    total*=n;
    long long index,indexx;
    float difference=0;
    long long counter= 0;
    for (int i=0;i<n;i++)
    {
        index=i*n;
        for (int j=i+1;j<m;j++)
        {
            indexx=index+j;
            difference=difference+abs(original[indexx]-recon[indexx]);
            counter+=1;
        }
    }
cout<<"\n the difference is: "<<difference/counter<<"\n";

}
void compute_q(float* dev_sigma, float* dev_cor,int Batch_size,int Length,int L,int wstep,char pri)//multiply matrix corr(b x b) to sigma(b x L) 
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    int info_gpu = 0;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_tau = NULL;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
     cudaMalloc ((void**)&d_tau, sizeof(float)*L);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    float* dev_y;
    cudaMalloc ((void**)&dev_y  , sizeof(float) *Batch_size * L);
    //cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle) ;
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error in creating cublas handle";
    }


    float* temp_sigma=new float[Batch_size*L];
    float* temp_result=new float[Batch_size*L];
    cudaMemcpy(temp_sigma, dev_sigma, sizeof(float) * Batch_size * L, cudaMemcpyDeviceToHost);
    const float alpha = 1.0;
    const float beta = 0.0;
     float* temp_cor=new float[Batch_size*Batch_size];
     stat = cublasSgemm(handle, CUBLAS_OP_N,  CUBLAS_OP_N, Batch_size, L, Batch_size,  &alpha, dev_cor , Batch_size, dev_sigma, Batch_size, &beta, dev_y, Batch_size);////multiplying sigma to correlation (dev_y)  y=sigma*cor


    if (stat != CUBLAS_STATUS_SUCCESS)
        {
            cout<<"error in cublasSgemmmmm";
        }
    cudaMemcpy(temp_result,dev_y, sizeof(float) * Batch_size * L, cudaMemcpyDeviceToHost);


//////////////////////////////////////
///////////////////QR finding Q///////////
// step 3: query working space of geqrf and orgqr
    cusolver_status = cusolverDnSgeqrf_bufferSize(
        cusolverH,
        Batch_size,
        L,
        dev_y,
        Batch_size,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusolver_status = cusolverDnSorgqr_bufferSize(
        cusolverH,
        Batch_size,
        L,
        L,
        dev_y,
        Batch_size,
         d_tau,
        &lwork_orgqr);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
 
  cusolver_status = cusolverDnSgeqrf(
        cusolverH,
        Batch_size,
        L,
        dev_y,
        Batch_size,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);


    // check if QR is successful or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

   // printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);




// step 5: compute Q
    cusolver_status= cusolverDnSorgqr(
        cusolverH,
        Batch_size,
        L,
        L,
        dev_y,
        Batch_size,
        d_tau,
        d_work,
        lwork,
        devInfo);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    //printf("after orgqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    float* Q=new float[Batch_size*L];
    cudaStat1 = cudaMemcpy(Q, dev_y, sizeof(float)*Batch_size*L, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    
/////////////////////////////////////
//////////////////////////////////////
//cout<<"\n ------------------Q------------------ \n";
/////////////////////////
/////////////Q is computed, compute B=Q^T x A
    float* h_B=new float[L*Batch_size];
    float* d_B;
    cudaMalloc ((void**)&d_B, sizeof(float)*L*Batch_size);

    stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, L, Batch_size,Batch_size,  &alpha, dev_y , Batch_size, dev_cor, Batch_size, &beta, d_B, L);////multiplying
     if (stat != CUBLAS_STATUS_SUCCESS)
         {
              cout<<"error in cublasSgemmmmm";
         }

     cudaMemcpy(h_B,d_B, sizeof(float) * Batch_size * L, cudaMemcpyDeviceToHost);

/////////////////////////Computing the difference between actual and reconstructed corr//////////////////////////
/*
    float* h_corr=new float[Batch_size*Batch_size];
    float* d_corr;
    cudaMalloc ((void**)&d_corr, sizeof(float)*Batch_size*Batch_size);

    stat = cublasSgemm(handle, CUBLAS_OP_N,  CUBLAS_OP_N, Batch_size, Batch_size,L,  &alpha, dev_y , Batch_size, d_B, L, &beta, d_corr, Batch_size);////multiplying

    if (stat != CUBLAS_STATUS_SUCCESS)
         {
             cout<<"error in cublasSgemmmmm";
         }

    cudaMemcpy(h_corr,d_corr, sizeof(float) * Batch_size * Batch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_cor, dev_cor, sizeof(float) * Batch_size * Batch_size, cudaMemcpyDeviceToHost);
    diff(temp_cor,h_corr,Batch_size,Batch_size);
*/
///////////////printeeee
     if(pri == 'y')
         {

                ofstream out;  ///text file
                string adr="./result/GPU_decom_Q" + to_string(wstep);
                out.open(adr);
		float tempo;
                for (long jjj=0;jjj<L*Batch_size;jjj++)
                {
                        out<< Q[jjj]<<"  ";
                }
                out.close();

                ofstream out2;
                adr="./result/GPU_decom_B" + to_string(wstep);

                out2.open(adr);


                for (long jjj=0;jjj<L*Batch_size;jjj++)
                {
                       out2<< h_B[jjj]<<"  ";
                        
                }
                out2.close();

       }
     cudaFree(dev_y);
     cudaFree(d_B);
     cudaFree(d_tau);
     cudaFree(devInfo);
     cudaFree(d_work);

}

