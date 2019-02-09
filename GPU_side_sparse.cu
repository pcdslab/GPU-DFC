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
//#include"preprocessing.cu"
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)
using namespace std;
long long remaining_N2(int , int ,long long );
long long remaining_N(int , int ,int );
void preprocessing(float * , int ,int );
long long remaining_mem(int , int ,int);
__global__ void ker_preprocessing(float*,float*,int ,int,int,int);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

long long get_avail_mem(int N,int L)
{
size_t free;
    size_t total_mem;
    cudaMemGetInfo(&free,&total_mem);
    long long available_mem = free;
    available_mem/=sizeof(float);
    available_mem-=(N*L*2);//Getting available memory without
return available_mem;

}

__global__ void ker_sparse(float * cormat,int n1,int n,float treshold)
{
    long idx = blockDim.x*blockIdx.x+threadIdx.x;
    long i = idx%n1;
    long j = idx/n1;

    if(i<=j || (i>j && cormat[j*n+i]<treshold) && i<n1 && j<n)
	cormat[j*n+i] = 0.0;



}

__global__ void ker2(float * cormat,int n1,int n,float treshold)
{
    long long idx = blockDim.x;
    idx*=blockIdx.x;
    idx+=threadIdx.x;
    long long kol = n1;
    kol *= n;
    if(idx <kol)
    {
        long i = idx/n1;
        long j = idx%n1;
        if(cormat[idx]<treshold)
	     cormat[idx] = 0;

	if(i<=j)
            cormat[idx] = 0;
    }

}

int CorMat_sparse(float * BOLD, int N, int L,int windowsize,int windowstep,float treshold,char pr)
{
   
    int slide_num=1;
    if(windowstep!=0)
         slide_num = (L-windowsize)/windowstep +1;
    cout<<" slide number is: "<<slide_num<<"\n";


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
    
//////////////////loop starts here////////
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


    //cout<<"step size: "<<wstep<<" ";

    ker_preprocessing<<<N,32,(windowsize+1)*sizeof(float)>>>(devBOLD,dev_windowedbold,N,L, windowsize, stepn);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();
    stepn+=windowstep ;

 

   cudaMemset(devCormat, 0, total*sizeof(float));    
//windowsize    
stat=  cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, N,N,windowsize,  &alpha, dev_windowedbold, windowsize, dev_windowedbold, windowsize, &beta, devCormat, N);
        cudaDeviceSynchronize();
    
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error performing multiplication";
        return stat;
    }
    
    
    int block_size=1024;//number of threads
    long long grid_size=1+((total-1)/block_size);//number of blocks

 ker_sparse<<<grid_size,block_size>>>(devCormat,N,N,treshold);//performing kernel for extracting and reordering correlations from upper triangle
   gpuErrchk( cudaPeekAtLastError() );



/////////////////////////////////Counting number of nozero per row

    cusparseStatus_t status;
    cusparseHandle_t sparsehandle=0;
    cusparseMatDescr_t descr=0;

    status = cusparseCreate(&sparsehandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "CUSPARSE Library initialization failed" << endl;
    }
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "Matrix descriptor initialization failed" << endl;
    }
    status = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cusparseSetMatType failed" << endl;
    }
    status = cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cusparseSetMatIndexBase failed" << endl;
    }


    int* nnzPerRow=0;
    cudaMalloc((void**)&nnzPerRow, sizeof(int)*N);
    int nnzTotal;
    //int* host_nrow=new int[N];
    status = cusparseSnnz(sparsehandle, CUSPARSE_DIRECTION_ROW, N, N, descr, devCormat, N, nnzPerRow, &nnzTotal);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "nnz calculation failed" << endl;
        cout << "status = " << status << endl;
    }


  // cout<<"\n number of nnz: "<<nnzTotal <<"\n";
    cusparseMatDescr_t descrX;
    cusparseCreateMatDescr(&descrX);
    float *csrValA;
    int *csrRowPtrA;
    int *csrColIndA;
     cudaMallocManaged( &csrValA, sizeof(float) * nnzTotal) ;
    cudaMallocManaged( &csrRowPtrA, sizeof(int) * (N+1)) ;
    cudaMallocManaged( &csrColIndA, sizeof(int) * nnzTotal) ;
    cusparseSdense2csr(sparsehandle, N, N,  descrX, devCormat,N, nnzPerRow, csrValA, csrRowPtrA, csrColIndA); 
    float *val=new float[nnzTotal];
    int* row=new int[N+1];
    int* column=new int[nnzTotal];
    cudaMemcpy(val,csrValA , sizeof(float)*nnzTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy( row, csrRowPtrA, sizeof(int)*(N+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(column, csrColIndA, sizeof(int)*nnzTotal, cudaMemcpyDeviceToHost);

    if(pr == 'y')
    {

                ofstream out;
                string adr="./result/val" + to_string(wstep);
		out.open(adr);
                ofstream out2;
		string adr2="./result/colptr" + to_string(wstep);
                out2.open(adr2);
                for (int jjj=0;jjj<nnzTotal;jjj++)
                {
                        out<< val[jjj]<<" ";
                        out2<<column[jjj]<<" ";
                }
                out.close();
  		out2.close();
		ofstream out3;
		string adr3="./result/rowptr" + to_string(wstep);
		out3.open(adr3);
		 for (int jjj=0;jjj<N+1;jjj++)
                {
			out3<<row[jjj]<<" ";
		}
                out3.close();
     }
    cudaFree (csrValA);
   cudaFree (csrRowPtrA);
     cudaFree(csrColIndA);


delete []val;
delete[]row;
delete[]column;



}//end of for step


/////////////////////////////////////

    cudaFree (devCormat);
//    cudaFree (dev_upper);
    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error in destroy";
        return stat;
    }


    return 1;
        
}


/*int CorMat_3(float * BOLD, int N, int L,int windowsize,int windowstep,float treshold,long long OOO)
{
    int slide_num=1;

    if(windowstep!=0)
         slide_num = (L-windowsize)/windowstep +1;
    cout<<" slide number is: "<<slide_num<<"\n";


    cudaError_t cudaStat;
    cublasStatus_t stat;

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
    cublasHandle_t handle;
    stat = cublasCreate(&handle) ;
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"Error in creating cublas handle";
        return stat;
    }


    const float alpha = 1.0;
    const float beta = 0.0;


    cusparseStatus_t status;
    cusparseHandle_t sparsehandle=0;
    cusparseMatDescr_t descr=0;

    status = cusparseCreate(&sparsehandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "CUSPARSE Library initialization failed" << endl;
    }
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "Matrix descriptor initialization failed" << endl;
    }
    status = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cusparseSetMatType failed" << endl;
    }
    status = cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cusparseSetMatIndexBase failed" << endl;
    }

   for(int wstep=0;wstep<slide_num;wstep++)//slide_num;wstep++)
  {
        cout<<wstep<<"   ";

        ker_preprocessing<<<N,32,(windowsize+1)*sizeof(float)>>>(devBOLD,dev_windowedbold,N,L, windowsize, stepn);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaDeviceSynchronize();
        stepn+=windowstep ;
        long long available_mem = get_avail_mem(N,L);
        int flag = 1;
        int block,N_prime;
        block=OOO;
        N_prime=N;
        long long temp2=0;
        int so_far=0;
        int pak=0;
        float* devCormat;
 //float* dev_upper;
        long long cormat_fullsize;
        int print_counter = 0;
        while(flag==1)
        {
        
        cout<<"this is block: "<<block<<"\n\n";        
            if(block==N_prime)//checking for the last chunk
                 flag=0;

             if(pak!=0)
                 {
                      cout<<"freed\n" ;
        //              cudaFree (dev_upper);
                      cudaFree (devCormat);///?????????????/

                  }
             cormat_fullsize=block;
             cormat_fullsize*=N_prime;
             cout<<block<<"  -  "<<N_prime<<"  - "<<devCormat;
            
             cudaStat=cudaMalloc ( (void**)&devCormat, sizeof(float) * cormat_fullsize) ;
             if (cudaStat != cudaSuccess)

             {
                cout<<"Error in Cuda Malloc and status is devcormat: "<<cudaStat;
                return cudaStat;
             } 

//  cout<<"\n IN PAK  0: "<<cormat_fullsize<<" *****";
            pak++;

            stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, block,N_prime,windowsize,  &alpha, dev_windowedbold+(so_far*windowsize), windowsize, dev_windowedbold + (so_far*windowsize), windowsize, &beta, devCormat, block);

//stat=  cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, N,N,windowsize,  &alpha, dev_windowedbold, windowsize, dev_windowedbold, windowsize, &beta, devCormat, N);

        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            cout<<"error in cublasSgemm, stat is: \n";
            cout<<stat<<"\n";
            return stat;
        }

        cudaDeviceSynchronize();

        temp2=block;
        temp2*=N_prime;

        int block_size=1024;
        long long grid_size=1+((temp2-1)/block_size);

        ker2<<<grid_size,block_size>>>(devCormat,block,N_prime,treshold);

        cudaDeviceSynchronize();
         so_far+=block;

        int* nnzPerRow=0;
        cudaMalloc((void**)&nnzPerRow, sizeof(int)*block);
        int nnzTotal;
        //int* host_nrow=new int[N];
         status = cusparseSnnz(sparsehandle, CUSPARSE_DIRECTION_ROW, block, N_prime, descr, devCormat, block, nnzPerRow, &nnzTotal);
         if (status != CUSPARSE_STATUS_SUCCESS) {
             cout << "nnz calculation failed" << endl;
             cout << "status = " << status << endl;
      }



    if(N_prime>block)
        {
             
            N_prime=N_prime-block;
            block=remaining_mem(N_prime,L,1);////remaining_N2( N_prime, L, available_mem);

            if(N_prime  <block)//checking last chunk
             block=N_prime;
             cout<<"YYYYYYYY "<<N_prime<<" "<<block<<" \n";;
        }



    cusparseMatDescr_t descrX;
    cusparseCreateMatDescr(&descrX);
    float *csrValA;
    int *csrRowPtrA;
    int *csrColIndA;
     cudaMallocManaged( &csrValA, sizeof(float) * nnzTotal) ;
    cudaMallocManaged( &csrRowPtrA, sizeof(int) * (N_prime+1)) ;
    cudaMallocManaged( &csrColIndA, sizeof(int) * nnzTotal) ;
    cusparseSdense2csr(sparsehandle, N_prime,block,  descrX, devCormat,N_prime, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
    float *val=new float[nnzTotal];
    int* row=new int[N_prime+1];
    int* column=new int[nnzTotal];
    cudaMemcpy(val,csrValA , sizeof(float)*nnzTotal, cudaMemcpyDeviceToHost);
    cudaMemcpy( row, csrRowPtrA, sizeof(int)*(N_prime+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(column, csrColIndA, sizeof(int)*nnzTotal, cudaMemcpyDeviceToHost);

*/
/*
    ofstream out;
    string adr="./result/val" +to_string(print_counter)+ to_string(wstep);
    out.open(adr);
    ofstream out2;
    string adr2="./result/colptr"+to_string(print_counter) + to_string(wstep);
    out2.open(adr2);
    for (int jjj=0;jjj<nnzTotal;jjj++)
         {
              out<< val[jjj]<<" ";
              out2<<column[jjj]<<" ";
         }
    out.close();
    out2.close();
    ofstream out3;
    string adr3="./result/rowptr"+to_string(print_counter) + to_string(wstep);
    out3.open(adr3);
    for (int jjj=0;jjj<N+1;jjj++)
        {
              out3<<row[jjj]<<" ";
        }

   print_counter++;
    out3.close();
*/
  /*  cudaFree (csrValA);
    cudaFree (csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(devCormat);

    }//flag

     cudaFree (devCormat);

}
return 1;

}*/
