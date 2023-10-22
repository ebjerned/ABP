#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
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

template <typename T> __global__ void assembleT(unsigned const int N,T* alpha, T*beta, T* val, int* col, int* row_ptr);

template <typename T, unsigned int blockSize> int lanczosAlg(unsigned const int N);
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


int main(int argc, char* argv[]){
    
    unsigned const int N = 100;
/*
    for(unsigned int i = 10; i < 250; i=i*1.1){
        lanczosAlg<double, 32>(i);
    }*/
        lanczosAlg<double, 32>(20);
    return 0;
}


template <typename T, unsigned int blockSize>
int lanczosAlg(unsigned const int N){

    int mode = 1;
    T *LAP_VAL_D, *I_VAL_D,*RES_VAL, *TERM1_VAL, *TERM2_VAL, *HLF_VAL, *FIN_VAL;
    int* LAP_COL_D, *LAP_ROW_D, *I_COL_D, *I_ROW_D, *RES_COL, *RES_ROW, *TERM1_COL, *TERM1_ROW, *TERM2_COL, *TERM2_ROW,*HLF_COL, *HLF_ROW, *FIN_COL, *FIN_ROW;

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


    assertCuda(cudaMalloc((void**) &FIN_VAL, 6*N*N*N*sizeof(T)));
    assertCuda(cudaMalloc((void**) &FIN_COL, 6*N*N*N*sizeof(int)));
    assertCuda(cudaMalloc((void**) &FIN_ROW, (N*N*N+1)*sizeof(int)));

    const auto lapb = std::chrono::steady_clock::now();
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

    //(T1+T2)+T3
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
    const double lape = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-lapb).count();

    double* CCS_VAL;
    int* CCS_COL, *CCS_CS, *CCS_CL;

    
    unsigned const int C = 32;
    int n_chunks = (int)ceil((float)N*N*N/C);

    dim3 convDim(1, n_chunks);



    assertCuda(cudaMalloc((void**) &CCS_CL, n_chunks*sizeof(int)));
    assertCuda(cudaMalloc((void**) &CCS_CS, (n_chunks+1)*sizeof(int)));
    assertCuda(cudaMalloc((void**) &CCS_VAL, (int)1.1*rec_size*sizeof(T))); // Account for padding
    assertCuda(cudaMalloc((void**) &CCS_COL, (int)1.1*rec_size*sizeof(int)));
    const auto cellcb = std::chrono::steady_clock::now(); 
    if (mode){
        prepCELLCSIG<<<1, blockDim, blockSize>>>(C, 1,N*N*N, FIN_ROW, CCS_CS, CCS_CL);
        CRStoCELLCSIG<T><<<convDim, blockDim>>>(C, 1, N*N*N, FIN_VAL, FIN_COL, FIN_ROW, CCS_VAL, CCS_COL, CCS_CS, CCS_CL);
    }
    const double cellce = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-cellcb).count();

    T* V, *Vp, *W, *A_vec, *B_vec, *a, *b;
    cudaMalloc((void**) &V, N*N*N*sizeof(T));
    cudaMalloc((void**) &Vp, N*N*N*sizeof(T));
    cudaMalloc((void**) &W, N*N*N*sizeof(T));
    cudaMalloc((void**) &A_vec, N*N*N*sizeof(T));
    cudaMalloc((void**) &B_vec, N*N*N*sizeof(T));
    cudaMalloc((void**) &a, sizeof(T));
    cudaMalloc((void**) &b, sizeof(T));


    T alpha, beta;


    const auto lanczosb = std::chrono::steady_clock::now();
//    const auto unitb = std::chrono::steady_clock::now();
    unitVec<T><<<kronDim2, blockDim>>>(N*N*N, 0, Vp);
//    double unite = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-unitb).count();
     
    const auto matb = std::chrono::steady_clock::now();
    if (mode){
        matvecCELLCSIGMA<T><<<kronDim2, blockDim>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, Vp, W);
    }else {
        matvecCRS<T><<<kronDim2, blockDim>>>(N*N*N, FIN_VAL, FIN_COL, FIN_ROW, Vp, W);

    }
    
    double mate = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-matb).count();

//    const auto dotb = std::chrono::steady_clock::now();
    dot<T><<<kronDim2, blockDim, blockSize>>>(N*N*N, W, Vp, &A_vec[0]);
  //  double dote = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-dotb).count();
    cudaMemcpy(&alpha, &A_vec[0], sizeof(T), cudaMemcpyDeviceToHost);

//    const auto linb = std::chrono::steady_clock::now();
    linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, Vp, W, 1, -alpha);
    //double line = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-linb).count();


    
    for(unsigned int j = 0; j < 20*N; j++){
        
  //      const auto sqb = std::chrono::steady_clock::now();
//        squareSum<T><<<kronDim2, blockDim, C>>>(N*N*N, W, &B_vec[j]);
        dot<T><<<kronDim2, blockDim, blockSize>>>(N*N*N, W, W, &B_vec[j]);
//        sqe += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-sqb).count();

        cudaMemcpy(&beta, &B_vec[j], sizeof(T), cudaMemcpyDeviceToHost);
        beta = sqrt(beta);
         
        if(beta != 0){
             //   const auto linb = std::chrono::steady_clock::now();
                linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, W, V, 0, 1/beta);
              //  line += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-linb).count();
        }else{
           // const auto unitb = std::chrono::steady_clock::now();
            unitVec<T><<<kronDim2, blockDim>>>(N*N*N, j, V);
          //  unite += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-unitb).count();
        }   

      //  const auto matb = std::chrono::steady_clock::now();
        if(mode){
            matvecCELLCSIGMA<T><<<kronDim2, blockDim>>>(N*N*N, C, CCS_VAL, CCS_COL, CCS_CS, CCS_CL, V,W);
        } else {
            matvecCRS<T><<<kronDim2, blockDim>>>(N*N*N, FIN_VAL, FIN_COL, FIN_ROW, V,W);
        }
        
    //    mate += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-matb).count();
        
        
        //const auto dotb = std::chrono::steady_clock::now();
        dot<T><<<kronDim2, blockDim, blockSize>>>(N*N*N, W, Vp, &A_vec[j]);
    //    dote += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-dotb).count();
        
        
        cudaMemcpy(&alpha, &A_vec[j], sizeof(T), cudaMemcpyDeviceToHost);
        
        
  //      const auto linb = std::chrono::steady_clock::now();
        linvec<T><<<kronDim2, blockDim>>>(N*N*N, V, Vp, V, -alpha, -beta);
        linvec<T><<<kronDim2, blockDim>>>(N*N*N, W, V, W, 1, 1);
//        line += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-linb).count();

    }


    const double lanczose = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-lanczosb).count();

    
    

//    assembleT<T><<<kronDim2, blockDim>>>(20*N ,A_vec, B_vec, FIN_VAL, FIN_COL, FIN_ROW);
    
    T *A, *B;
    A = (T*) malloc(20*N*sizeof(T));
    B = (T*) malloc((20*N-1)*sizeof(T));
    //assertCuda(cudaMemcpy(A, A_vec, 20*N*sizeof(T), cudaMemcpyDeviceToHost));
    //assertCuda(cudaMemcpy(B, B_vec, (20*N-1)*sizeof(T), cudaMemcpyDeviceToHost));


//    printVector<T>(20*N-1, A);

//    printCRS_matrix(20*N, val, col, row);
//    std::cout << "N:\tLaplacian\tCRStoCELLCS\tUnitVec\tMatVec\tDot\tLinOp\tSquareSum\tLanczosTOT" <<std::endl;
    
    double BW =0; 
    if(mode){
            // VAL, COL, CS, CL, VEC, RES
         BW = 1e-9*20*N*((rec_size+14*N*N*N)*sizeof(T)+(2*n_chunks+N*N*N)*sizeof(int))/lanczose;
    }else{
            // VAL, COL, ROW, VEC, RES
         BW = 1e-9*20*N*((rec_size+14*N*N*N)*sizeof(T)+(rec_size+N*N*N)*sizeof(int))/lanczose;
    }
    //std::cout << N <<"\t"<< lape << "\t" << cellce << /*"\t" <<unite <<*/ "\t" << mate << "\t" << BW << /*"\t" << dote << "\t"<< line << "\t" << sqe <<*/ "\t" << lanczose << std::endl;
    std::cout << N << "," << rec_size  << "," << lape << ","  << cellce << "," << BW << "," << lanczose << std::endl;
//    std::cout << N << "," << rec_size << "," << lape << ", " << cellce << "," << unite << "," << mate << "," << dote << "," << line << "," << BW << lanczose << std::endl;
    cudaFree(W);
    cudaFree(V);
    cudaFree(Vp);
    cudaFree(A_vec);
    cudaFree(B_vec);
    cudaFree(FIN_VAL);
    cudaFree(FIN_COL);
    cudaFree(FIN_ROW);
    cudaFree(CCS_VAL);
    cudaFree(CCS_COL);
    cudaFree(CCS_CS);
    cudaFree(CCS_CL);
    cudaFree(a);
    cudaFree(b);
    free(A);
    free(B);

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
    for(int i = 0; i <M; ++i){
        int offset = 0;
        int row_ptr = row_ptrC[tidy*M+i];
        for(int l = row_ptrA[tidy]; l < row_ptrA[tidy+1]; ++l){
            int Acol = colA[l];
            T Aval = valA[l];
            for( int j = row_ptrB[i]; j < row_ptrB[i+1]; ++j){
                T entry = Aval*valB[j];
                int col = Acol*M +colB[j];
                valC[row_ptr+offset] = entry;
                colC[row_ptr+offset] = col;
                offset += 1;
            }   
        }    
    }       
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
                ic++;
                ia++;
                
            } else if (colA[A_row_ptr+ia] == colB[B_row_ptr+ib]){
                ic++;
                ia++;
                ib++;
        
            } else{
                ic++;
                ib++;
            }
        
        }
        while(ia < (row_ptrA[tid+1]-row_ptrA[tid])){
            ic++;
            ia++;
        }
        while(ib < (row_ptrB[tid+1]-row_ptrB[tid])){
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
    int ic =  row_ptrC[tid];
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
    }
}
template <typename T>
__global__ void CRStoCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, T* val, int* col, int* row_ptr, T* val_o,int* col_o, int* cs, int* cl){
    unsigned int tid = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int bid = blockIdx.y;
        for(unsigned int k = 0; k < cl[bid]; ++k){
                if((k > (row_ptr[tid+1]-row_ptr[tid]-1)) || (tid+1) > N){
                    val_o[cs[bid]+k*C+threadIdx.y] = 0.0;
                    col_o[cs[bid]+k*C+threadIdx.y] = col_o[cs[bid]+threadIdx.y];


                } else{

                    val_o[cs[bid]+k*C+threadIdx.y] = val[row_ptr[tid]+k];
                    col_o[cs[bid]+k*C+threadIdx.y] = col[row_ptr[tid]+k];
               }
        }

    
    
    return;
}

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
}


template <typename T>
void printCRS_matrix(unsigned const int N, T* val, int* col, int* row_ptr){
    printf("Row:\tCol:\tVal:\n");
    for(int i = 0; i < N; ++i){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; ++j){
            std::cout <<"("<< row_ptr[i] << "\t" << col[j] << "\t" << val[j] << ")" << std::endl; 
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



template <typename T>
__global__ void assembleT(unsigned const int N,T* alpha, T*beta, T* val, int* col, int* row_ptr){
    int r_index = (blockDim.y*blockIdx.y + threadIdx.y);
    
    int diag = r_index;
    if (diag >= N) return;
    if(diag > 0){
        val[diag*3-1] = beta[diag-1];
        col[diag*3-1] = diag-1;
    }
    val[diag*3] = alpha[diag];
    col[diag*3] = diag;
    if(diag < (N-1)){
        val[diag*3+1] = beta[diag+1];
        col[diag*3+1] = diag+1;
    }
    
    row_ptr[diag] = (diag == 0) ? 0 : 3*diag-1;
    
    if(diag==N-1){
        row_ptr[N] = 3*N-2;
    }
    return;
}
