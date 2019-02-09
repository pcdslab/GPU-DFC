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
using namespace std;

long long remaining_N2(int , int ,long long );
long long remaining_N(int , int ,int );
/////void preprocessing(float * , int ,int );
/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

*/
__global__ void ker_preprocessing(float* bold,float* dev_windowbold,int n,int m,int windowsize,int step)//step: ta koja oumadi jelo where is the start point
{

   extern __shared__ float sh[];
    //long long idx=blockDim.x*blockIdx.x+threadIdx.x;
    float avg=0;
    int slide =1+((windowsize-1)/32);// 1+((m-1)/32);//ceil(m/32);
    long start_idx = blockIdx.x * m + step;
    int index_shared=threadIdx.x;
    int counter=0;
    for(int i=0;i<slide;i++)
        {
            if(counter<windowsize && index_shared<windowsize)//m)
                {
                sh[index_shared]=bold[start_idx+threadIdx.x];
                index_shared+=32;
                start_idx+=32;
                counter+=32;
                }
           __syncthreads();

        }
    index_shared=threadIdx.x;



for(int i=0;i<slide;i++)
        {
            if(index_shared<windowsize)//m)
            {
                avg = avg + sh[index_shared];
                index_shared += 32;
            }
        }
        __syncthreads();

for (int i=16; i>0; i=i/2)
    avg += __shfl_down(avg, i);


if(threadIdx.x==0)
{
sh[windowsize]=avg/windowsize;
//printf("\n im in block: %d  %f \n",blockIdx.x,sh[windowsize]);
}
 __syncthreads();
 index_shared=threadIdx.x;
avg=0;

for(int i=0;i<slide;i++)
        {
            if(index_shared<windowsize)//m)
            {
                avg+=(sh[index_shared]-sh[windowsize])*(sh[index_shared]-sh[windowsize]);
                sh[index_shared]=sh[index_shared]-sh[windowsize];
                index_shared += 32;
            }
        }
  __syncthreads();

for (int i=16; i>0; i=i/2)
    avg += __shfl_down(avg, i);

if(threadIdx.x==0)
{

 sh[windowsize]=sqrtf(avg);
}


counter=0;
     start_idx =blockIdx.x * windowsize;// blockIdx.x * m + step;
   index_shared=threadIdx.x;
    for(int i=0;i<slide;i++)
        {
            if(counter<windowsize&&index_shared<windowsize/*m*/&& start_idx+threadIdx.x<(blockIdx.x+1) * windowsize /*blockIdx.x * m +step+windowsize*/){//m){
       /////////         //bold[start_idx+threadIdx.x]=sh[index_shared]/sh[windowsize];//sh[m];
                dev_windowbold[start_idx+threadIdx.x]=sh[index_shared]/sh[windowsize];
                index_shared+=32;
                start_idx+=32;
                counter+=32;
                }
           __syncthreads();

        }


}

