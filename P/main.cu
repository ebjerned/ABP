#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#define assertCuda(error_code) \
    if(error_code != cudaSuccess){\
        std::cout << "The cuda call in " << __FILE__ << " on line " \
        << __LINE__ << "resulted in the errro '" \
        << cudaGetErrorString(error_code) << "'" << std::endl; \
        std::abort();\
    }\

__global__ void prepCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, int* row_ptr, int* cs_o, int* cl_o);
template <typename T>
__global__ void CRStoCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, T* val, int* col, int* row_ptr, T* val_o,int* col_o, int* cs_o, int* cl_o);
template <typename T>
__global__ void assemble1DLaplace(unsigned const int N, T* val, int* col, int* row_ptr);

template <typename T> __global__ void set_vector(const int N, const T val, T* vec);
template <typename T> __global__ void assembleIdentity(unsigned const int N, T* val, int* col, int* row_ptr);
__global__ void prepKroecker(unsigned const int N, unsigned const int M, int* row_ptrA, int* row_ptrB, int* row_ptrC); 
template <typename T> __global__ void kronecker(unsigned const int N, unsigned const int M, T* valA, int* colA, int* row_ptrA, T* valB, int* colB, int* row_ptrB, T* valC, int* colC, int* row_ptrC);
__global__ void prepAddCRS(unsigned const int N, unsigned const int M, int* colA, int* row_ptrA, int* colB,  int* row_ptrB, int* row_ptrC);
template <typename T> __global__ void addCRS(unsigned const int N, unsigned const int M, T* valA, int* colA, int* row_ptrA, T* valB, int* colB, int* row_ptrB, T* valC, int* colC, int* row_ptrC);

template <typename T> __global__ void lanczos(); // Return T matrix, in either CRS or CELLCSIGMA
template <typename T> __global__ void matvecCRS(unsigned const int N, T* valA, int* colA, int* row_ptrA, T* vec, T* res);
template <typename T> __global__ void matvecCELLCSIGMA(unsigned const int N, unsigned const int C, T* valCCS, int* colCCS, int* cs, int* cl, T * vec, T* res);
template <typename T> __global__ void dot(unsigned const int N, T* valA, T* valB, T* res); 
template <typename T> __global__ void squareSum(unsigned int N, T* vec, T* res);
template <typename T> __global__ void unitVec(unsigned const int N, unsigned const int pos, T* res); // Remake set_vector()


template <typename T, unsigned int blockSize> int assemble3DLaplace(unsigned const int N, T* RES_VAL, int* RES_COL, int* RES_ROW);
template <typename T> int CRStoCCS(unsigned const int N, unsigned const int C, unsigned const int sigma, T* valCRS, int* colCRS, int* rowCRS, T* valCCS, int* colCCS, int* csCCS, int* clCCS);
template <typename T>__global__ void linvec(unsigned int N, T* vecA, T* vecB, T* res, const T alpha, const T beta);
template <typename T> void printVector(unsigned const int N, T* vec);


template <typename T> void printCRS_matrix(unsigned const int N, T* val, int* col, int* row_ptr);
template <typename T> void printCELLCSIG_matrix(unsigned const int N, unsigned const int C, T* valCCS, int* colCCS, int* cs, int* cl);

template <typename T>
__global__ void set_vector(const int N, const T val, T* vec){
    const int idx_base = threadIdx.y + blockIdx.y*blockDim.y;
    if(idx_base < N) vec[idx_base] = val;

}


template <typename T>
void test(unsigned const int N){
   
    T* val;
    T* RES_VAL_D, *CCS_VAL;
    int* col;
    int* row_ptr;
    int* RES_COL_D, *CCS_COL;
    int* RES_ROW_D, *CCS_ROW;
    int* CCS_CS,* CCS_CL;
    T* vector;
    dim3 gridDim(1, 4);
    dim3 blockDim(1,32);

   dim3 gridDim2(1,313);
    unsigned int C = 32;
    unsigned int KRONECKER_SIZE = N*N*N;
    unsigned int CELLCSIGMA_CHUNKS = (int)ceil((float)KRONECKER_SIZE/C);
/*
    cudaMalloc((void**) &cl_device, ((int)ceil((float)N*N/32))*sizeof(int));
    cudaMalloc((void**) &cs_device, ((int)ceil((float)N*N/32+1))*sizeof(int));
    cudaMalloc((void**) &val4_device, 5*N*N*sizeof(T));
    cudaMalloc((void**) &col4_device, 5*N*N*sizeof(int));
*/
    cudaMalloc((void**) &vector, CELLCSIGMA_CHUNKS*32*sizeof(T));
    

    T* test;
    T norm;
    assertCuda(cudaMalloc((void**) &test, sizeof(T)));
//   assemble1DLaplace<<<gridDim, blockDim>>>(N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D);
//   assembleIdentity<<<gridDim, blockDim>>>(N, I_VAL_D, I_COL_D, I_ROW_D);
   dim3 convDim(1, CELLCSIGMA_CHUNKS);
    //set_vector<T><<<gridDim2, blockDim>>>(N*N, 1.0, vector);
    //squareSum<T><<<dim3(1,((int)ceil(N*N/32.f))), dim3(1,32)>>>(N*N, vector, test);
    //assertCuda(cudaGetLastError());
    //assertCuda(cudaMemcpy(&norm, test, sizeof(T), cudaMemcpyDeviceToHost));
    //printf("%lf\n", norm);
//    double* vec = (double*) malloc(32*CELLCSIGMA_CHUNKS*sizeof(double));
//    cudaMemcpy(vec, vector, 32*CELLCSIGMA_CHUNKS*sizeof(double), cudaMemcpyDeviceToHost);
//    printVector(N*N, vec);
//    matvecCRS<<<gridDim, blockDim>>>(N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, vector, RES_VAL_D);

   
   /*
    cudaMalloc((void**), &V, N*N*N*sizeof(T));
    cudaMalloc((void**), &Vp, N*N*N*sizeof(T));
    cudaMalloc((void**), &W, N*N*N*sizeof(T));
    cudaMalloc((void**), &A_vec, N*N*N*sizeof(T));
    cudaMalloc((void**), &B_vec, (N*N*N-1)*sizeof(T));*/
//    cudaMalloc((void**) &RES_ROW_D, (N*N*N+1)*sizeof(int));
    assemble3DLaplace<T, 32>(N, RES_VAL_D, RES_COL_D, RES_ROW_D);
    int vec;// = (int*)malloc(N*N*N*sizeof(int));
  //  assertCuda(cudaMemcpy(&vec, &RES_ROW_D[N*N*N], sizeof(int), cudaMemcpyDeviceToHost));
 //   printf("%i\n", vec);
//    printVector<int>(N, vec);
   // CRStoCCS<T>(N*N*N, C, 1, RES_VAL_D, RES_COL_D, RES_ROW_D, CCS_VAL, CCS_COL, CCS_CS, CCS_CL);
    /*



    unit_vector<T><<<>>>(N*N*N, j, Vp);
    matvecCELLCSIGMA<T><<<>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, Vp, W);
    T* a;
    T alpha;
    assertCuda(cudaMalloc((void**) &a, sizeof(T)));
    dot<T><<<>>>(N*N*N, W, Vp, a);
    assertCuda(cudaMemcpy(&alpha, a, sizeof(T), cudaMemcpyDeviceToHost));
    linvec<T><<<>>>(N*N*N, W, Vp, W, 1, -alpha);
    A_vec[0] = alpha;
//Start LOOP

    T* b;
    T beta;
    assertCuda(cudaMalloc((void**) &a, sizeof(T)));
    squaredSum<T><<<>>>(N*N*N, W, b);
    assertCuda(cudaMemcpy(&beta, b, sizeof(T), cudaMemcpyDeviceToHost));
    
    if(beta != 0){
        linvec<T><<<>>>(N*N*N, W, W, V, 0, 1/beta);
    }else{
        unit_vector<T><<<>>>(N*N*N, j, V);
    }

    matvecCELLCSIGMA<T><<<>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, W, V);

    dot<T><<<>>>(N*N*N, W, Vp, a);
    assertCuda(cudaMemcpy(&alpha, a, sizeof(T), cudaMemcpyDeviceToHost));
    linvec<T><<<>>>(N*N*N, V, Vp, V, -alpha, -beta);
    linvec<T><<<>>>(N*N*N, W, V, W, 1, 1);

    A_vec[j] = alpha;
    B_vec[j] = beta;*/

// END lOOp
   
   //prepCELLCSIG<<<1, blockDim,C>>>(C, 1, N*N, RES_ROW_D, cs_device, cl_device);
   //CRStoCELLCSIG<<<convDim, blockDim>>>(C, 1, N*N, RES_VAL_D, RES_COL_D, RES_ROW_D, val4_device, col4_device, cs_device, cl_device);
    


//    matvecCELLCSIGMA<<<convDim, blockDim>>>(N*N, C, val4_device, col4_device, cs_device, cl_device, vector, RES_VAL_D);
//   prepAddCRS<<<gridDim, blockDim>>>(N,N, LAP_COL_D, LAP_ROW_D, I_COL_D,  I_COL_D, RES_ROW_D);
//   addCRS<<<gridDim, blockDim,N+1>>>(N, N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, I_VAL_D, I_COL_D, I_ROW_D, RES_VAL_D, RES_COL_D, RES_ROW_D);

    cudaDeviceSynchronize();

    int rec_size;
    cudaMemcpy(&rec_size, &RES_ROW_D[N*N], sizeof(int), cudaMemcpyDeviceToHost);
//    rec_size = N*N;
    printf("Rec-size: %i\n", rec_size);
    row_ptr = (int*) malloc((N*N+1)*sizeof(int));
    cudaMemcpy(row_ptr, RES_ROW_D, (N*N+1)*sizeof(int), cudaMemcpyDeviceToHost);  
    val = (T*) malloc(rec_size*sizeof(T));
    col = (int*) malloc(rec_size*sizeof(int));
    cudaMemcpy(val, RES_VAL_D, rec_size*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(col, RES_COL_D, rec_size*sizeof(int), cudaMemcpyDeviceToHost);
    

    int rec_size2;
    cudaMemcpy(&rec_size2, &CCS_CS[CELLCSIGMA_CHUNKS], sizeof(int), cudaMemcpyDeviceToHost);
    int* cs = (int*) malloc(CELLCSIGMA_CHUNKS*sizeof(int));
    cudaMemcpy(cs, CCS_CS, CELLCSIGMA_CHUNKS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    rec_size2 = N*N;
    printf("Rec-size: %i\n", rec_size2);
    T* valCCS = (T*) malloc(rec_size2*sizeof(T));
    int* colCCS = (int*) malloc(rec_size2*sizeof(int));
    int* cl = (int*) malloc(CELLCSIGMA_CHUNKS*sizeof(int));
    cudaMemcpy(valCCS, CCS_VAL, rec_size2*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(colCCS, CCS_COL, rec_size2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cl, CCS_CL, CELLCSIGMA_CHUNKS*sizeof(int), cudaMemcpyDeviceToHost);

  //  printCRS_matrix(N*N, val, col,row_ptr);
//    printCELLCSIG_matrix(N*N, C, valCCS, colCCS, cs, cl);
   // printVector<double>(N*N, val);
 //   cudaFree(LAP_VAL_D);
 //   cudaFree(LAP_COL_D);
   // cudaFree(LAP_ROW_D);
   // cudaFree(I_VAL_D);
   // cudaFree(I_COL_D);
   // cudaFree(I_ROW_D);
    cudaFree(RES_VAL_D);
    cudaFree(RES_COL_D);
    cudaFree(RES_ROW_D);
    cudaFree(CCS_VAL);
    cudaFree(CCS_COL);
    cudaFree(CCS_CS);
    cudaFree(CCS_CL);
    free(val);
    free(col);
    free(row_ptr);
    free(cs);
    free(cl);
    free(colCCS);
    free(valCCS);

    
    
}




int main(int argc, char* argv[]){
    
    unsigned const int N = 100;

    test<double>(N);
    return 0;
}


template <typename T, unsigned int blockSize>
int assemble3DLaplace(unsigned const int N, T* FIN_VAL, int* FIN_COL, int* FIN_ROW){

    T *LAP_VAL_D, *I_VAL_D,*RES_VAL, *TERM1_VAL, *TERM2_VAL, *HLF_VAL;
    int* LAP_COL_D, *LAP_ROW_D, *I_COL_D, *I_ROW_D, *RES_COL, *RES_ROW, *TERM1_COL, *TERM1_ROW, *TERM2_COL, *TERM2_ROW,*HLF_COL, *HLF_ROW;

    dim3 gridDim(1,(int)ceil((float)N/blockSize));
    dim3 kronDim1(1,(int)ceil((float)N*N/blockSize));
    dim3 kronDim2(1,(int)ceil((float)N*N*N/blockSize));
    dim3 kronDim3(1,(int)ceil((float)N*N*N/blockSize));
    dim3 blockDim(1, blockSize);

    cudaMalloc((void**) &LAP_VAL_D, (3*N-2)*sizeof(T));
    cudaMalloc((void**) &LAP_COL_D, (3*N-2)*sizeof(int));
    cudaMalloc((void**) &LAP_ROW_D, (N+1)*sizeof(int)); 

    cudaMalloc((void**) &I_VAL_D, N*sizeof(T));
    cudaMalloc((void**) &I_COL_D, N*sizeof(int));
    cudaMalloc((void**) &I_ROW_D, (N+1)*sizeof(int)); 
   
    cudaMalloc((void**) &RES_VAL, 3*N*N*sizeof(T));
    cudaMalloc((void**) &RES_COL, 3*N*N*sizeof(int));
    cudaMalloc((void**) &RES_ROW, (N*N+1)*sizeof(int));

    cudaMalloc((void**) &TERM1_VAL, 3*N*N*N*sizeof(T));
    cudaMalloc((void**) &TERM1_COL, 3*N*N*N*sizeof(int));
    cudaMalloc((void**) &TERM1_ROW, (N*N*N+1)*sizeof(int));

    cudaMalloc((void**) &TERM2_VAL, 3*N*N*N*sizeof(T));
    cudaMalloc((void**) &TERM2_COL, 3*N*N*N*sizeof(int));
    cudaMalloc((void**) &TERM2_ROW, (N*N*N+1)*sizeof(int));
    
    cudaMalloc((void**) &HLF_VAL, 6*N*N*N*sizeof(T));
    cudaMalloc((void**) &HLF_COL, 6*N*N*N*sizeof(int));
    cudaMalloc((void**) &HLF_ROW, (N*N*N+1)*sizeof(int));


    cudaMalloc((void**) &FIN_VAL, 6*N*N*N*sizeof(T));
    cudaMalloc((void**) &FIN_COL, 6*N*N*N*sizeof(int));
    cudaMalloc((void**) &FIN_ROW, (N*N*N+1)*sizeof(int));

    assemble1DLaplace<T><<<gridDim, blockDim>>>(N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D);
    assembleIdentity<T><<<gridDim, blockDim>>>(N, I_VAL_D, I_COL_D, I_ROW_D);

    //T1: kron(kron(L, I), I)
   prepKroecker<<<kronDim1, blockDim>>>(N, N, LAP_ROW_D, I_ROW_D, RES_ROW);
   kronecker<T><<<kronDim1, blockDim>>>(N, N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, I_VAL_D, I_COL_D, I_ROW_D, RES_VAL, RES_COL, RES_ROW);
   prepKroecker<<<kronDim2, blockDim>>>(N*N, N, RES_ROW, I_ROW_D, TERM1_ROW);
   kronecker<T><<<kronDim2, blockDim>>>(N*N, N, RES_VAL, RES_COL, RES_ROW, I_VAL_D, I_COL_D, I_ROW_D, TERM1_VAL, TERM1_COL, TERM1_ROW);

    //T2: kron(kron(I,L),I)

   prepKroecker<<<kronDim1, blockDim>>>(N, N, I_ROW_D, LAP_ROW_D, RES_ROW);
   kronecker<T><<<kronDim1, blockDim>>>(N, N, I_VAL_D, I_COL_D, I_ROW_D, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, RES_VAL, RES_COL, RES_ROW);
   prepKroecker<<<kronDim2, blockDim>>>(N*N, N, RES_ROW, I_ROW_D, TERM2_ROW);
   kronecker<T><<<kronDim2, blockDim>>>(N*N, N, RES_VAL, RES_COL, RES_ROW, I_VAL_D, I_COL_D, I_ROW_D, TERM2_VAL, TERM2_COL, TERM2_ROW);

    //T1 + T2
    prepAddCRS<<<kronDim2, blockDim>>>(N*N*N, N*N*N, TERM1_COL, TERM1_ROW, TERM2_COL, TERM2_ROW, HLF_ROW);
   addCRS<T><<<kronDim2, blockDim>>>(N*N*N, N*N*N, TERM1_VAL, TERM1_COL, TERM1_ROW, TERM2_VAL, TERM2_COL, TERM2_ROW, HLF_VAL, HLF_COL, HLF_ROW);

    cudaFree(TERM2_VAL);
    cudaFree(TERM2_COL);
    cudaFree(TERM2_ROW);


    //T3: kron(kron(I,I),L)
   prepKroecker<<<kronDim1, blockDim>>>(N, N, I_ROW_D, I_ROW_D, RES_ROW);
   kronecker<T><<<kronDim1, blockDim>>>(N, N, I_VAL_D, I_COL_D, I_ROW_D, I_VAL_D, I_COL_D, I_ROW_D, RES_VAL, RES_COL, RES_ROW);
   prepKroecker<<<kronDim2, blockDim>>>(N*N, N, RES_ROW, I_ROW_D, TERM1_ROW);
   kronecker<T><<<kronDim2, blockDim>>>(N*N, N, RES_VAL, RES_COL, RES_ROW, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, TERM1_VAL, TERM1_COL, TERM1_ROW);

    cudaFree(LAP_VAL_D);
    cudaFree(LAP_COL_D);
    cudaFree(LAP_ROW_D);
    cudaFree(I_VAL_D);
    cudaFree(I_COL_D);
    cudaFree(I_ROW_D);
    cudaFree(RES_VAL);
    cudaFree(RES_COL);
    cudaFree(RES_ROW);

    prepAddCRS<<<kronDim2, blockDim>>>(N*N*N, N*N*N, HLF_COL, HLF_ROW, TERM1_COL, TERM1_ROW, FIN_ROW);
    addCRS<T><<<kronDim2, blockDim>>>(N*N*N, N*N*N, HLF_VAL, HLF_COL, HLF_ROW, TERM1_VAL, TERM1_COL, TERM1_ROW, FIN_VAL, FIN_COL, FIN_ROW);

    cudaFree(HLF_VAL);
    cudaFree(HLF_COL);
    cudaFree(HLF_ROW);
    cudaFree(TERM1_VAL);
    cudaFree(TERM1_COL);
    cudaFree(TERM1_ROW);

    int rec_size;
    assertCuda(cudaMemcpy(&rec_size, &FIN_ROW[N*N*N], sizeof(int), cudaMemcpyDeviceToHost));
    printf("Laplace size: %i\n", rec_size);


    double* CCS_VAL;
    int* CCS_COL, *CCS_CS, *CCS_CL;

    
    unsigned const int C = 32;
    int n_chunks = (int)ceil((float)N*N*N/C);

    dim3 convDim(1, n_chunks);


    assertCuda(cudaMalloc((void**) &CCS_CL, n_chunks*sizeof(int)));
    assertCuda(cudaMalloc((void**) &CCS_CS, (n_chunks+1)*sizeof(int)));
    assertCuda(cudaMalloc((void**) &CCS_VAL, (int)1.1*rec_size*sizeof(T))); // Account for padding
    assertCuda(cudaMalloc((void**) &CCS_COL, (int)1.1*rec_size*sizeof(int)));
    prepCELLCSIG<<<1, blockDim, C>>>(C, 1,N*N*N, FIN_ROW, CCS_CS, CCS_CL);
    CRStoCELLCSIG<T><<<convDim, blockDim>>>(C, 1, N*N*N, FIN_VAL, FIN_COL, FIN_ROW, CCS_VAL, CCS_COL, CCS_CS, CCS_CL);

    
//    CRStoCCS<T>(N*N*N, C, 1, FIN_VAL, FIN_COL, FIN_ROW, CCS_VAL, CCS_COL, CCS_CS, CCS_CL);

    int mode = 1;
    T* V, *Vp, *W, *A_vec, *B_vec;

    cudaMalloc((void**) &V, N*N*N*sizeof(T));
    cudaMalloc((void**) &Vp, N*N*N*sizeof(T));
    cudaMalloc((void**) &W, N*N*N*sizeof(T));
    cudaMalloc((void**) &A_vec, N*N*N*sizeof(T));
    cudaMalloc((void**) &B_vec, N*N*N*sizeof(T));
    cusparseConstSpMatDescr_t cuSpares_mat;
    unitVec<T><<<kronDim2, blockDim>>>(N*N*N, 0, Vp);
    if (mode==2){
        matvecCELLCSIGMA<T><<<kronDim2, blockDim>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, Vp, W);
    }else if (mode ==1){
        cusparseCreateConstCsr(cuSpares_mat, N*N*N, N*N*N, rec_size, FIN_ROW, FIN_COL, FIN_VAL, CUSPARSE_INDEX_32I, CUSPARESE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,1,  cuSpares_mat, cuSpares_vec, 0, out, CUDA_R_&¤F)
    
    } else{
    
        matvecCRS<T><<<kronDim2, blockDim>>>(N*N*N, FIN_VAL, FIN_COL, FIN_ROW, Vp, W);
    }
    
    T* a;
    T alpha;
    assertCuda(cudaMalloc((void**) &a, sizeof(T)));
    dot<T><<<kronDim2, blockDim,C>>>(N*N*N, W, Vp, a);
    assertCuda(cudaMemcpy(&A_vec[0], &a[0], sizeof(T), cudaMemcpyDeviceToDevice));
    assertCuda(cudaMemcpy(&alpha, &a[0], sizeof(T), cudaMemcpyDeviceToHost));

    linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, Vp, W, 1, -alpha);
//    A_vec[0] = alpha;
//Start LOOP
    printf("Starting loop\n");
    for(unsigned int j = 0; j < 20*N; j++){
        



        T* b;
        T beta;
        assertCuda(cudaMalloc((void**) &b, sizeof(T)));
        squareSum<T><<<kronDim2, blockDim, C>>>(N*N*N, W, b);
        assertCuda(cudaMemcpy(&beta, b, sizeof(T), cudaMemcpyDeviceToHost));
        assertCuda(cudaMemcpy(&B_vec[j], b, sizeof(T), cudaMemcpyDeviceToDevice));
        
        if(beta != 0){
            linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, W, V, 0, 1/beta);
        }else{
            unitVec<T><<<kronDim2, blockDim>>>(N*N*N, j, V);
        }   

        if(mode){
            matvecCELLCSIGMA<T><<<kronDim2, blockDim>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, V,W);
        } else {
            matvecCRS<T><<<kronDim2, blockDim>>>(N*N*N, FIN_VAL, FIN_COL, FIN_ROW, V,W);
        }
        dot<T><<<kronDim2, blockDim, C>>>(N*N*N, W, Vp, a);
        assertCuda(cudaMemcpy(&A_vec[j], a, sizeof(T), cudaMemcpyDeviceToDevice));
        assertCuda(cudaMemcpy(&alpha, a, sizeof(T), cudaMemcpyDeviceToHost));
        linvec<T><<<kronDim2, blockDim>>>(N*N*N, V, Vp, V, -alpha, -beta);
        linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, V, W, 1, 1);

        printf("%i\n", j);

}
// END lOOp

    return rec_size;

}


template <typename T> 
int CRStoCCS(unsigned const int N, unsigned const int C, unsigned const int sigma, T* valCRS, int* colCRS, int* rowCRS, T* valCCS, int* colCCS, int* csCCS, int* clCCS){
    dim3 blockDim(1,C); 
    int n_chunks = (int)ceil((float)N/C);
    dim3 convDim(1, n_chunks);
    int rec_size = 5959999;
//    assertCuda(cudaMemcpy(&rec_size, &rowCRS[N], sizeof(int), cudaMemcpyDeviceToHost));
    assertCuda(cudaMalloc((void**) &clCCS, n_chunks*sizeof(int)));
    assertCuda(cudaMalloc((void**) &csCCS, (n_chunks+1)*sizeof(int)));
    assertCuda(cudaMalloc((void**) &valCCS, (int)1.1*rec_size*sizeof(T))); // Account for padding
    assertCuda(cudaMalloc((void**) &colCCS, (int)1.1*rec_size*sizeof(int)));
    prepCELLCSIG<<<1, blockDim, C>>>(C, sigma, N, rowCRS, csCCS, clCCS);
    CRStoCELLCSIG<T><<<convDim, blockDim>>>(C, sigma, N, valCRS, colCRS, rowCRS, valCCS, colCCS, csCCS, clCCS);
    
    
    assertCuda(cudaMemcpy(&rec_size, &csCCS[n_chunks], sizeof(int), cudaMemcpyDeviceToHost));
    return rec_size;
}

template <typename T>
__global__ void assemble1DLaplace(unsigned const int N, T* val, int* col, int* row_ptr){
    int r_index = (blockDim.y*blockIdx.y + threadIdx.y);
    
    T c = (1/(T)((N+1)*(N+1)));
    int diag = r_index;
    if (diag >= N) return;
    if(diag > 0){
        val[diag*3-1] = c*1;
        col[diag*3-1] = diag-1;
    }
    val[diag*3] = c*(-2);
    col[diag*3] = diag;
    if(diag < (N-1)){
        val[diag*3+1] = c*1;
        col[diag*3+1] = diag+1;
    }
    
    row_ptr[diag] = (diag == 0) ? 0 : 3*diag-1;
    
    if(diag==N-1){
        row_ptr[N] = 3*N-2;
        printf("%i\n", row_ptr[N]); 
    }
    return;
}
template <typename T>
__global__ void assembleIdentity(unsigned const int N, T* val, int* col, int* row_ptr){
    int r_index = (blockDim.y*blockIdx.y +threadIdx.y);
    if (r_index >= N) return;
    val[r_index] = 1;
    col[r_index] = r_index;
    row_ptr[r_index] = r_index;

    if(r_index == N-1) row_ptr[N] = N;
}

__global__ void prepKroecker(unsigned const int N, unsigned const int M, int* row_ptrA, int* row_ptrB, int* row_ptrC){ 
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;

    if(tidy >= N) return;
    //TODO: find non iterative way to compute pointers. Should be possible as it A and B and the number of elements per row in C is possible to compute

    if(tidy == 0){
        row_ptrC[0] = 0;
        for(unsigned int k = 0; k < N; ++k){
            for(unsigned int i = 0; i < M; ++i){
                row_ptrC[k*M+i+1] = row_ptrC[k*M+i] + (row_ptrA[k+1] -row_ptrA[k])*(row_ptrB[i+1]-row_ptrB[i]);
           }
       }
       //printf("C: %i\n", row_ptrC[M*N]);
    }

}
template <typename T>
__global__ void kronecker(unsigned const int N, unsigned const int M, T* valA, int* colA, int* row_ptrA, T* valB, int* colB, int* row_ptrB, T* valC, int* colC, int* row_ptrC){

    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if(tidy >= N) return;
    
    /*
    //TODO: find non iterative way to compute pointers. Should be possible as it A and B and the number of elements per row in C is possible to compute

    if(tidy == 0 && tidx == 0){
        row_ptrC[0] = 0;
        for(unsigned int k = 0; k < N; ++k)
            for(unsigned int i = 0; i < M; ++i)
                row_ptrC[k*M+i+1] = row_ptrC[k*M+i] + (row_ptrA[k+1] -row_ptrA[k])*(row_ptrB[i+1]-row_ptrB[i]);
           
        printf("%i\n", row_ptrC[N*M]);
    }*/
    __syncthreads();
//    if(tid == 0){
     //   row_ptrC[0] = 0;
     //   for(int k = 0; k < N; ++k){
        //    int A_row_ptr = row_ptrA[k];
       //     int n_elementsA = row_ptrA[k+1] - row_ptrA[k];
       for(int i = 0; i <M; ++i){
                int offset = 0;
         //       int n_elementsB = row_ptrB[i+1] - row_ptrB[i];
           //     row_ptrC[k*M+i+1] = (row_ptrC[k*M+i]+n_elementsB*n_elementsA);
                int row_ptr = row_ptrC[tidy*M+i];
       //         printf("(%i,%i) %i\n",tidy,tidx, row_ptrA[tidy+1]-row_ptrA[tidy]);
                for(int l = row_ptrA[tidy]; l < row_ptrA[tidy+1]; ++l){
                    int Acol = colA[l];
                    T Aval = valA[l];
                    for( int j = row_ptrB[i]; j < row_ptrB[i+1]; ++j){
                        T entry = Aval*valB[j];
                        int col = Acol*M +colB[j];
 //                       printf("%i\t%i\t%lf\n", row_ptrC[tidy*M+tidx], col, entry);            
                        valC[row_ptr+offset] = entry;
                        colC[row_ptr+offset] = col;
                        offset += 1;
                    }   
                }    
            }       
       // }
//    }
    /*
    return;*/
}
__global__ void prepAddCRS(unsigned const int N, unsigned const int M, int* colA, int* row_ptrA, int* colB,  int* row_ptrB, int* row_ptrC){
    
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    if(tid >= N) return;
    row_ptrC[tid] = 0;

    int ia = 0;
    int ib = 0;
    int ic =  0;
    int A_row_ptr = row_ptrA[tid];
    int B_row_ptr = row_ptrB[tid];

    //TODO: Could store the to-be-added values of A and B to C into temp fast storage, such that it isn't needed to be fetched from large A and B when
    // consolidating the row_ptrs after the synch.
    while((ia < (row_ptrA[tid+1]-row_ptrA[tid])) && (ib < (row_ptrB[tid+1] -row_ptrB[tid]))){
                
            if(colA[A_row_ptr+ia] < colB[B_row_ptr+ib]){
//                colC[ic] = colA[A_row_ptr+ia];
 //               valC[ic] = valA[A_row_ptr+ia];
                ic++;
                ia++;
                
            } else if (colA[A_row_ptr+ia] == colB[B_row_ptr+ib]){
//                colC[ic] = colA[A_row_ptr+ia];
//                valC[ic] = valA[A_row_ptr+ia] + valB[B_row_ptr+ib];
                ic++;
                ia++;
                ib++;
        
            } else{
//                colC[ic] = colB[B_row_ptr+ib];
//                valC[ic] = valB[B_row_ptr+ib];
                ic++;
                ib++;
            }
        
        }
        while(ia < (row_ptrA[tid+1]-row_ptrA[tid])){
//            colC[ic] = colA[A_row_ptr+ia];
//            valC[ic] = valA[A_row_ptr+ia];
            ic++;
            ia++;
        }
        while(ib < (row_ptrB[tid+1]-row_ptrB[tid])){
//            colC[ic] = colA[B_row_ptr+ib];
//            valC[ic] = valA[B_row_ptr+ib];
            ic++;
            ib++;
        }

        row_ptrC[tid+1] = ic;


    
    if(tid ==0){
        for(unsigned int i = 1; i <= N; ++i){
            row_ptrC[i] = row_ptrC[i] + row_ptrC[i-1];
 //           printf("%i\n", row_ptrC[i]);
        }

    }

}

template <typename T>
__global__ void addCRS(unsigned const int N, unsigned const int M, T* valA, int* colA, int* row_ptrA, T* valB, int* colB, int* row_ptrB, T* valC, int* colC, int* row_ptrC){

    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    if(tid >= N) return;

    int ia = 0;
    int ib = 0;
    int ic =  0;
 /*   int A_row_ptr = row_ptrA[tid];
    int B_row_ptr = row_ptrB[tid];

    //TODO: Could store the to-be-added values of A and B to C into temp fast storage, such that it isn't needed to be fetched from large A and B when
    // consolidating the row_ptrs after the synch.
    while((ia < (row_ptrA[tid+1]-row_ptrA[tid])) && (ib < (row_ptrB[tid+1] -row_ptrB[tid]))){
                
            if(colA[A_row_ptr+ia] < colB[B_row_ptr+ib]){
//                colC[ic] = colA[A_row_ptr+ia];
 //               valC[ic] = valA[A_row_ptr+ia];
                ic++;
                ia++;
                
            } else if (colA[A_row_ptr+ia] == colB[B_row_ptr+ib]){
//                colC[ic] = colA[A_row_ptr+ia];
//                valC[ic] = valA[A_row_ptr+ia] + valB[B_row_ptr+ib];
                ic++;
                ia++;
                ib++;
        
            } else{
//                colC[ic] = colB[B_row_ptr+ib];
//                valC[ic] = valB[B_row_ptr+ib];
                ic++;
                ib++;
            }
        
        }
        while(ia < (row_ptrA[tid+1]-row_ptrA[tid])){
//            colC[ic] = colA[A_row_ptr+ia];
//            valC[ic] = valA[A_row_ptr+ia];
            ic++;
            ia++;
        }
        while(ib < (row_ptrB[tid+1]-row_ptrB[tid])){
//            colC[ic] = colA[B_row_ptr+ib];
//            valC[ic] = valA[B_row_ptr+ib];
            ic++;
            ib++;
        }
    
        offsets[tid+1] = ic;
    __syncthreads();*/

    
    __syncthreads();
//    printf("%i\n", row_ptrC[tid]);
    ia = 0;
    ib = 0;
    ic =  row_ptrC[tid];
    int A_row_ptr = row_ptrA[tid];
    int B_row_ptr = row_ptrB[tid];
    while((ia < (row_ptrA[tid+1]-row_ptrA[tid])) && (ib < (row_ptrB[tid+1] -row_ptrB[tid]))){
            if(colA[A_row_ptr+ia] < colB[B_row_ptr+ib]){
                colC[ic] = colA[A_row_ptr+ia];
                valC[ic] = valA[A_row_ptr+ia];
                ic++;
                ia++;
                
            } else if (colA[A_row_ptr+ia] == colB[B_row_ptr+ib]){
                colC[ic] = colA[A_row_ptr+ia];
                valC[ic] = valA[A_row_ptr+ia] + valB[B_row_ptr+ib];
                ic++;
                ia++;
                ib++;
        
            } else if(colA[A_row_ptr+ia] > colB[B_row_ptr+ib]){
                colC[ic] = colB[B_row_ptr+ib];
                valC[ic] = valB[B_row_ptr+ib];
                ic++;
                ib++;
            }
    }
        while(ia < (row_ptrA[tid+1]-row_ptrA[tid])){
            colC[ic] = colA[A_row_ptr+ia];
            valC[ic] = valA[A_row_ptr+ia];
            ic++;
            ia++;
        }
        while(ib < (row_ptrB[tid+1]-row_ptrB[tid])){
            colC[ic] = colB[B_row_ptr+ib];
            valC[ic] = valB[B_row_ptr+ib];
            ic++;
            ib++;
        }
    

}

__global__ void prepCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, int* row_ptr, int* cs_o, int* cl_o){
    extern __shared__ int max_vec[];
    unsigned int limit = (int)ceil((float)N/C);
    if(threadIdx.y == 0) cs_o[0] = 0;
    for(unsigned int i = 0; i <limit; ++i){
        unsigned int tid = blockDim.y*i + threadIdx.y;
        int max = 0;
        max_vec[threadIdx.y] = 0;
        if(tid < N){
//            printf("%i\n", row_ptr[tid]);
            max_vec[threadIdx.y] = ((row_ptr[tid+1]-row_ptr[tid]) > max) ? row_ptr[tid+1]-row_ptr[tid] : max;
        }
        __syncthreads();
        for(unsigned int j = 0; j < C; ++j){
            max = max_vec[j] > max ? max_vec[j] : max;
        }
         
        if(threadIdx.y==0){
            cl_o[i] = max;
            cs_o[i+1] = cl_o[i]*C + cs_o[i];
        }
//        if(threadIdx.y == 0) printf("%i\n", cs_o[i]);
    }


}
template <typename T>
__global__ void CRStoCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, T* val, int* col, int* row_ptr, T* val_o,int* col_o, int* cs, int* cl){
/*    cs_o[0] = 0;
   if(threadIdx.y == 0 && threadIdx.x == 0){ 
    unsigned int limit = (int)ceil((float)N/C)*C;
    for(unsigned int i = 0; i <limit; i+=C){
        unsigned int max = 0;
        printf("%i\n", limit);
        printf("Chunk: %i\n", i);
        for(unsigned int j = 0; j < C; ++j){
            if(i+j >= N) break;
            max = ((row_ptr[i+j+1]-row_ptr[i+j]) > max) ? row_ptr[i+j+1]-row_ptr[i+j] : max;
        }

        for(unsigned int k = 0; k < max; ++k){
            for(unsigned int l = 0; l < C; l++){
                if((k > (row_ptr[i+l+1]-row_ptr[i+l]-1)) || (i+l+1) > N){
                    printf("Padded at %i\n", cs_o[i/32]+k*C+l);
                    val_o[cs_o[i/32]+k*C+l] = 0.0;
                    col_o[cs_o[i/32]+k*C+l] = 0;


                } else{

                    val_o[cs_o[i/32]+k*C+l] = val[row_ptr[i+l]+k];
                    col_o[cs_o[i/32]+k*C+l] = col[row_ptr[i+l]+k];
                    }
            }
        }
        cl_o[i/32] = max;
        cs_o[i/32+1] = cl_o[i/32]*C + cs_o[i/32];
        printf("\t\tCl: %i, Cs: %i\n", cl_o[i/32], cs_o[i/32+1]);

    }

    }
    return;
    */
//    cs_o[0] = 0;
//    extern __shared__ int max_vec[];
 //   unsigned int limit = (int)ceil((float)N/C)*C;
    unsigned int tid = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int bid = blockIdx.y;
        for(unsigned int k = 0; k < cl[bid]; ++k){
                if((k > (row_ptr[tid+1]-row_ptr[tid]-1)) || (tid+1) > N){
//                    printf("Padded at %i\n", cs_o[bid]+k*C+threadIdx.y);
                    val_o[cs[bid]+k*C+threadIdx.y] = 0.0;
                    col_o[cs[bid]+k*C+threadIdx.y] = col_o[cs[bid]+threadIdx.y];


                } else{

                    val_o[cs[bid]+k*C+threadIdx.y] = val[row_ptr[tid]+k];
                    col_o[cs[bid]+k*C+threadIdx.y] = col[row_ptr[tid]+k];
               }
        }

        if(tid == 0) printf("\t\tCl: %i, Cs: %i\n", cl[bid], cs[bid+1]);
    
    
    return;
}

/*
__global__ void lanczos(){


    T w  = 0;

    for(unsigned int i = rowA[tid]; i < rowA[tid+1]; ++i){
        w += valA[i]*vec[colA[i]];
    }



}
*/


template <typename T>
__global__ void matvecCRS(unsigned const int N, T* valA, int* colA, int* row_ptrA, T* vec, T* res){
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    if(tid >= N) return;
    T tmp = 0;
    for(unsigned int i = row_ptrA[tid]; i < row_ptrA[tid+1]; ++i){
        tmp += valA[i]*vec[colA[i]];
    }
    res[tid] = tmp;
}

template <typename T>
__global__ void dot(unsigned const int N, T* valA, T* valB, T* res){ 
    // TODO: Improve reduction algorithm
    extern __shared__ T data[];
    int tid = blockIdx.y*blockDim.y + threadIdx.y;
    int ty = threadIdx.y;
    if(tid >= N) return;
    data[ty] = valA[tid]*valB[tid];
    if(tid==0) *res = 0;
    __syncthreads();
    for(unsigned int s = 1; s < blockDim.y; s *=2){
        if((ty %(2* s)==0) &&((tid+s) < N)){
            data[ty] += data[ty+s];
        }
        __syncthreads();
    }
    if(ty == 0){ 
        atomicAdd(res, data[0]);              
    }

}
template <typename T>
__global__ void matvecCELLCSIGMA(unsigned const int N, unsigned const int C, T* valCCS, int* colCCS, int* cs, int* cl, T* vec, T* res){
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    int bid = blockIdx.y;
    if(tid >= N) return;
    T tmp = 0;
    for(unsigned int i = 0; i < cl[bid]; ++i){
        tmp += valCCS[cs[bid]+i*C+threadIdx.y]*vec[colCCS[cs[bid]+i*C+threadIdx.y]];
    }
    res[tid] = tmp;
//    if(tid==0) printf("%i\n", cs[N]);
}


template <typename T>
void printCRS_matrix(unsigned const int N, T* val, int* col, int* row_ptr){
    printf("Row:\tCol:\tVal:\n");
    for(int i = 0; i < N; ++i){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; ++j){
            std::cout <<"("<< row_ptr[i] << "\t" << col[j] << "\t" << val[j] << ")" << std::endl; 
//            printf("(%i\t%i\t%lf)\n",row_ptr[i], col[j], val[j]);

        }
    
    }
    return;
}

template <typename T>
void printCELLCSIG_matrix(unsigned const int N, unsigned const int C, T* valCCS, int* colCCS, int* cs, int* cl){
   
    for(int i = 0; i < (int)ceil((float)N/C); ++i){
        printf("Chunk start: %i, Chunk width: %i\n", cs[i], cl[i]);
        for(int l = 0; l < C; ++l){
            for(int j = 0; j < cl[i]; ++j){
                std::cout << "[" << colCCS[cs[i]+j*C+l] << ",\t" << valCCS[cs[i]+j*C+l] << std::endl;
//                printf("[%i,\t%lf] ",colCCS[cs[i]+j*C+l], valCCS[cs[i]+j*C+l]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return;
}

template<typename T>
void printVector(unsigned const int N, T* vec){
    for(unsigned int i = 0; i < N; ++i)
        std::cout << vec[i] << std::endl;

}
template <typename T>
__global__ void linvec(unsigned int N, T* vecA, T* vecB, T* res, const T alpha, const T beta){
    int tid = blockIdx.y*blockDim.y + threadIdx.y;
    if(tid >= N) return;
    res[tid] = alpha*vecA[tid] + beta*vecB[tid];
}


template <typename T>
__global__ void squareSum(unsigned int N, T* vec, T* res){
    //TODO: Improve reduction algorithm to better from E1.
    extern __shared__ T data[];
    int tid = blockIdx.y*blockDim.y + threadIdx.y;
    int ty = threadIdx.y;
    if(tid >= N) return;
    data[ty] = vec[tid]*vec[tid];
    if(tid==0) *res = 0;
    __syncthreads();
    for(unsigned int s = 1; s < blockDim.y; s *=2){
        if((ty %(2* s)==0) &&((tid+s) < N)){
            data[ty] += data[ty+s];
        }
        __syncthreads();
    }
    if(ty == 0){ 
        atomicAdd(res, data[0]);              

    }
}


template <typename T>
__global__ void unitVec(unsigned const int N, unsigned const int pos, T* res){
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    if(tid < N){
        T c = 1;
        res[tid] = (tid == pos) ? c : 0;
    }
}
