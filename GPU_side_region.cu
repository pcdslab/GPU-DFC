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
#include <thrust/device_vector.h> 
#include <thrust/count.h>
#include <thrust/execution_policy.h> 
#include <cusparse.h>
#include<cusparse_v2.h>
#include<cuda_runtime.h>
#include"preprocessing.cu"
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)
using namespace std;
long long remaining_N2(int , int ,long long );
long long remaining_N(int , int ,int );
void preprocessing(float * , int ,int );
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void ker(float* __restrict__ cormat,float* __restrict__ upper,int n,int wstep)
{
    long idx = blockDim.x*blockIdx.x+threadIdx.x;
    long i = idx%n;
    long j = idx/n;
    long long index=j;
    index*=n;
    index+=i;

    if(i<j && i<n && j<n)
    {
        long tmp=i;
        tmp*=(i+1);
        tmp/=2;
        long tmp_2=i;
        tmp_2*=n;
        tmp_2=tmp_2-tmp;
       tmp_2+=j;
       tmp_2-=i;

	upper[tmp_2-1]=cormat[j*n+i];
    }


}


__global__ void ker2(float * cormat, float * upper,int n1,int n,long long upper_size,int N,int i_so_far,long long M1)
{
long long idx = blockDim.x;
idx*=blockIdx.x;
idx+=threadIdx.x;
long i = idx/n;
long j = idx%n;

if(i<j && i<n1 && j<n)// &&i<N &&j<N && idx<(n1*n))
{
        long long tmp=i;
        tmp*=(i+1);
        tmp/=2;
        long long tmp_2=i;
        tmp_2*=n;
        tmp_2=tmp_2-tmp;
        tmp_2+=j;
        tmp_2-=i;
        long long indexi=n1;
        indexi*=j;
        indexi=indexi+i;
        upper[tmp_2-1]=cormat[indexi];

}

}


int CorMat_2( float * BOLD, int N, int L,int windowsize,int windowstep,char pr)
{

    	int slide_num;
    	if(windowstep!=0)
        	slide_num = ((L-windowsize)/windowstep) +1;
    	cout<<" slide number is: "<<slide_num<<"\n";
    	long long M1 = (N-1); //computing the  size of correlaion matrix
    	M1 *= N;
    	M1 /= 2;
	float* dev_upper;
        cudaMalloc ((void**)&dev_upper, sizeof(float) * M1 * slide_num) ;
        float* dev_upper_temp = dev_upper;
	float* cpucorrupper=new float[M1*slide_num];
    	cudaError_t cudaStat;
    	cublasStatus_t stat;
    	long long total=N;//size of total correlation matrix
    	total*=N;
    	float * total_cormat = new float [total];
    	float * devBOLD; //Allocating space in GPU for storing fMRI data
	cudaStat = cudaMalloc ((void**)&devBOLD, sizeof(float) * L * N) ;

    	if (cudaStat != cudaSuccess)
    	{
        	cout<<"Error in Cuda Malloc";
        	return cudaStat;
    	}
    	float* dev_windowedbold;
    	long tul=N;
    	tul*=windowsize;
    	cudaStat = cudaMalloc ((void**)&dev_windowedbold, sizeof(float) * tul) ;

    	if (cudaStat != cudaSuccess)
    	{
        	cout<<"Error in Cuda Malloc";
        	return cudaStat;
    	}

    	cudaMemcpy( devBOLD,BOLD, sizeof(float) *N*L, cudaMemcpyHostToDevice);
    
//////////////////loop starts here////////
	int stepn=0;
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
    		stat=cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, N,N,windowsize,  &alpha, dev_windowedbold, windowsize, dev_windowedbold, windowsize, &beta, devCormat, N);
	        cudaDeviceSynchronize();
    
	    	if (stat != CUBLAS_STATUS_SUCCESS)
    		{
		        cout<<"Error performing multiplication";
		        return stat;
    		}
    
    		int block_size=1024;//number of threads
	    	long long grid_size=1+((total-1)/block_size);//number of blocks
	  	ker<<<grid_size,block_size>>>(devCormat,dev_upper_temp,N,wstep);//performing kernel for extracting and reordering correlations from upper triangle
	   	gpuErrchk( cudaPeekAtLastError() );
		dev_upper_temp += M1;

	}//end of for step
          
        cudaMemcpy(cpucorrupper, dev_upper, sizeof(float)*slide_num*M1, cudaMemcpyDeviceToHost);
       

   if (pr == 'y')
   {
       long long inddd = 0;
        for (int iii =0;iii<slide_num;iii++)
	{
        	ofstream out;
		string adr="./result/corr_step_" + to_string(iii);
		
		out.open(adr);
		for (int jjj=0;jjj<M1;jjj++)
		{
			out<< cpucorrupper[inddd]<<"  ";
			inddd++;
		}
		out.close();
        }
   }     
    return 1;
}
