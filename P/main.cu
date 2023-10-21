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




typedef struct CRS_matrix{
    double* _val;
    int* _col;
    int* _row_ptr;
    int _nnz;
    int _m;

} CRS_matrix;

typedef struct CELLCSIG_matrix{
    int* _cl;
    int* _cs;
    double*_val;

} CELLCSIG_matrix;



__global__ void prepCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, int* row_ptr, int* cs_o, int* cl_o);
__global__ void CRStoCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, double* val, int* col, int* row_ptr, double* val_o,int* col_o, int* cs_o, int* cl_o);
__global__ void assemble1DLaplace(unsigned const int N, double* val, int* col, int* row_ptr);
__global__ void set_vector(const int N, const double val, double* vec);
__global__ void assembleIdentity(unsigned const int N, double* val, int* col, int* row_ptr);
__global__ void prepKroecker(unsigned const int N, unsigned const int M, int* row_ptrA, int* row_ptrB, int* row_ptrC); 
__global__ void kronecker(unsigned const int N, unsigned const int M, double* valA, int* colA, int* row_ptrA, double* valB, int* colB, int* row_ptrB, double* valC, int* colC, int* row_ptrC);
__global__ void prepAddCRS(unsigned const int N, unsigned const int M, int* colA, int* row_ptrA, int* colB,  int* row_ptrB, int* row_ptrC);
__global__ void addCRS(unsigned const int N, unsigned const int M, double* valA, int* colA, int* row_ptrA, double* valB, int* colB, int* row_ptrB, double* valC, int* colC, int* row_ptrC);

__global__ void assemble3DLaplace();
__global__ void lanczos(); // Return T matrix, in either CRS or CELLCSIGMA
__global__ void matvecCRS(unsigned const int N, double* valA, int* colA, int* row_ptrA, double* vec, double* res);
__global__ void matvecCELLCSIGMA(unsigned const int N, unsigned const int C, double* valCCS, int* colCCS, int* cs, int* cl, double * vec, double* res);
__global__ void addvec(double* vecA, double* vecB, double* res); // Maybe subract aswell
__global__ void multvec(double* vecA, double scalar, double* res);
__global__ void euclideanNorm(double* vec, double res);
__global__ void unitVec(int length, int pos, int* res); // Remake set_vector()




template<typename T> void printVector(unsigned const int N, T* vec);
void printCRS_matrix(unsigned const int N, double* val, int* col, int* row_ptr);
void printCELLCSIG_matrix(unsigned const int N, unsigned const int C, double* valCCS, int* colCCS, int* cs, int* cl);
void allocateCRS(unsigned const int N, CRS_matrix* mat);
void freeCRS(CRS_matrix* mat);
__global__ void set_vector(const int N, const double val, double* vec){
    const int idx_base = threadIdx.y + blockIdx.y*blockDim.y;
    if(idx_base < N) vec[idx_base] = val;

}


void test(unsigned const int N){
   
    double* val;
    double* LAP_VAL_D, *I_VAL_D, *RES_VAL_D, *val4_device;
    int* col;
    int* row_ptr;
    int* LAP_COL_D, *I_COL_D, *RES_COL_D, *col4_device;
    int* LAP_ROW_D, *I_ROW_D, *RES_ROW_D;
    int* cl_device,* cs_device;
    double* vector;
    dim3 gridDim(1, 4);
    dim3 blockDim(1,32);

   dim3 gridDim2(1,313);
    unsigned int C = 32;
    unsigned int KRONECKER_SIZE = N*N;
    unsigned int CELLCSIGMA_CHUNKS = (int)ceil((float)KRONECKER_SIZE/C);

   // val = (double*)malloc(3*N*sizeof(double));
    //col = (int*)malloc(3*N*sizeof(int));
//    row_ptr = (int*)malloc((N+1)*sizeof(int));
    
   cudaMalloc((void**) &LAP_VAL_D, 3*N*sizeof(double));
   cudaMalloc((void**) &LAP_COL_D, 3*N*sizeof(int));
   cudaMalloc((void**) &LAP_ROW_D, (N+1)*sizeof(int)); 

   cudaMalloc((void**) &I_VAL_D, N*sizeof(double));
   cudaMalloc((void**) &I_COL_D, N*sizeof(int));
   cudaMalloc((void**) &I_ROW_D, (N+1)*sizeof(int)); 
   
   cudaMalloc((void**) &RES_VAL_D, 3*N*N*sizeof(double));
   cudaMalloc((void**) &RES_COL_D, 3*N*N*sizeof(int));
   cudaMalloc((void**) &RES_ROW_D, (N*N+1)*sizeof(int));


    cudaMalloc((void**) &cl_device, ((int)ceil((float)N*N/32))*sizeof(int));
    cudaMalloc((void**) &cs_device, ((int)ceil((float)N*N/32+1))*sizeof(int));
    cudaMalloc((void**) &val4_device, 5*N*N*sizeof(double));
    cudaMalloc((void**) &col4_device, 5*N*N*sizeof(int));

    cudaMalloc((void**) &vector, CELLCSIGMA_CHUNKS*32*sizeof(double));
    


   assemble1DLaplace<<<gridDim, blockDim>>>(N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D);
   assembleIdentity<<<gridDim, blockDim>>>(N, I_VAL_D, I_COL_D, I_ROW_D);
   dim3 convDim(1, CELLCSIGMA_CHUNKS);
    set_vector<<<gridDim2, blockDim>>>(N*N, 1.0, vector);
//    double* vec = (double*) malloc(32*CELLCSIGMA_CHUNKS*sizeof(double));
//    cudaMemcpy(vec, vector, 32*CELLCSIGMA_CHUNKS*sizeof(double), cudaMemcpyDeviceToHost);
//    printVector(N*N, vec);
//    matvecCRS<<<gridDim, blockDim>>>(N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, vector, RES_VAL_D);

   
   prepKroecker<<<gridDim2, blockDim>>>(N, N, LAP_ROW_D, I_ROW_D, RES_ROW_D);
   kronecker<<<gridDim2, blockDim>>>(N, N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, I_VAL_D, I_COL_D, I_ROW_D, RES_VAL_D, RES_COL_D, RES_ROW_D);
   prepCELLCSIG<<<1, blockDim,C>>>(C, 1, N*N, RES_ROW_D, cs_device, cl_device);
   CRStoCELLCSIG<<<convDim, blockDim>>>(C, 1, N*N, RES_VAL_D, RES_COL_D, RES_ROW_D, val4_device, col4_device, cs_device, cl_device);
    


    matvecCELLCSIGMA<<<convDim, blockDim>>>(N*N, C, val4_device, col4_device, cs_device, cl_device, vector, RES_VAL_D);
//   prepAddCRS<<<gridDim, blockDim>>>(N,N, LAP_COL_D, LAP_ROW_D, I_COL_D,  I_COL_D, RES_ROW_D);
//   addCRS<<<gridDim, blockDim,N+1>>>(N, N, LAP_VAL_D, LAP_COL_D, LAP_ROW_D, I_VAL_D, I_COL_D, I_ROW_D, RES_VAL_D, RES_COL_D, RES_ROW_D);

    cudaDeviceSynchronize();

    int rec_size;
    cudaMemcpy(&rec_size, &RES_ROW_D[N*N], sizeof(int), cudaMemcpyDeviceToHost);
    rec_size = N*N;
    printf("Rec-size: %i\n", rec_size);
    row_ptr = (int*) malloc((N*N+1)*sizeof(int));
    cudaMemcpy(row_ptr, RES_ROW_D, (N*N+1)*sizeof(int), cudaMemcpyDeviceToHost);  
    val = (double*) malloc(rec_size*sizeof(double));
    col = (int*) malloc(rec_size*sizeof(int));
    cudaMemcpy(val, RES_VAL_D, rec_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(col, RES_COL_D, rec_size*sizeof(int), cudaMemcpyDeviceToHost);
    

    int rec_size2;
    cudaMemcpy(&rec_size2, &cs_device[CELLCSIGMA_CHUNKS], sizeof(int), cudaMemcpyDeviceToHost);
    int* cs = (int*) malloc(CELLCSIGMA_CHUNKS*sizeof(int));
    cudaMemcpy(cs, cs_device, CELLCSIGMA_CHUNKS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    rec_size2 = N*N;
    printf("Rec-size: %i\n", rec_size2);
    double* valCCS = (double*) malloc(rec_size2*sizeof(double));
    int* colCCS = (int*) malloc(rec_size2*sizeof(int));
    int* cl = (int*) malloc(CELLCSIGMA_CHUNKS*sizeof(int));
    cudaMemcpy(valCCS, val4_device, rec_size2*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(colCCS, col4_device, rec_size2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cl, cl_device, CELLCSIGMA_CHUNKS*sizeof(int), cudaMemcpyDeviceToHost);

  //  printCRS_matrix(N*N, val, col,row_ptr);
   // printCELLCSIG_matrix(N*N, C, valCCS, colCCS, cs, cl);
    printVector<double>(N*N, val);
    cudaFree(LAP_VAL_D);
    cudaFree(LAP_COL_D);
    cudaFree(LAP_ROW_D);
    cudaFree(I_VAL_D);
    cudaFree(I_COL_D);
    cudaFree(I_ROW_D);
    cudaFree(RES_VAL_D);
    cudaFree(RES_COL_D);
    cudaFree(RES_ROW_D);
    cudaFree(val4_device);
    cudaFree(col4_device);
    cudaFree(cl_device);
    cudaFree(cs_device);
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

    test(N);
    return 0;
}

void allocateCRS(unsigned const int N, CRS_matrix* mat){
    assertCuda(cudaMalloc((void**)&mat, sizeof(CRS_matrix)));
    assertCuda(cudaMalloc((void**)mat->_val, 3*N*sizeof(double)));
    assertCuda(cudaMalloc((void**)&(mat->_col), 3*N*sizeof(int)));
    assertCuda(cudaMalloc((void**)&(mat->_row_ptr), N*sizeof(int)));
    assertCuda(cudaMalloc((void**)&(mat->_nnz), sizeof(int)));
    assertCuda(cudaMalloc((void**)&(mat->_m), sizeof(int)));
}

void freeCRS(CRS_matrix* mat){
    cudaFree(mat->_val);
    cudaFree(mat->_col);
    cudaFree(mat->_row_ptr);
    cudaFree(&(mat->_nnz));
    cudaFree(&(mat->_m));
    cudaFree(mat);
}



/*
__global__ void assemble3DLaplace(unsigned const int N, double* val, int* col, int* row_ptr){
    assemble1DLaplace(N, val, col, row_ptr);

    return;
}
*/
__global__ void assemble1DLaplace(unsigned const int N, double* val, int* col, int* row_ptr){
    int r_index = (blockDim.y*blockIdx.y + threadIdx.y);
    
    double c = (1/(double)((N+1)*(N+1)));
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

__global__ void assembleIdentity(unsigned const int N, double* val, int* col, int* row_ptr){
    int r_index = (blockDim.y*blockIdx.y +threadIdx.y);
    if (r_index >= N) return;
    val[r_index] = 1;
    col[r_index] = r_index;
    row_ptr[r_index] = r_index;

    if(r_index == N-1){ row_ptr[N] = N;

        printf("%i\n", row_ptr[N]); }
}
__global__ void prepKroecker(unsigned const int N, unsigned const int M, int* row_ptrA, int* row_ptrB, int* row_ptrC){ 
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("test2");
    if(tidy >= N || tidx >= M) return;
    //TODO: find non iterative way to compute pointers. Should be possible as it A and B and the number of elements per row in C is possible to compute

    if(tidy == 0 && tidx == 0){
        row_ptrC[0] = 0;
        for(unsigned int k = 0; k < N; ++k)
            for(unsigned int i = 0; i < M; ++i)
                row_ptrC[k*M+i+1] = row_ptrC[k*M+i] + (row_ptrA[k+1] -row_ptrA[k])*(row_ptrB[i+1]-row_ptrB[i]);
           
    //    printf("%i\n", row_ptrC[N*M]);
    }

}
__global__ void kronecker(unsigned const int N, unsigned const int M, double* valA, int* colA, int* row_ptrA, double* valB, int* colB, int* row_ptrB, double* valC, int* colC, int* row_ptrC){
    //printf("test");

    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    if((tidy >= N) || (tidx >= M)){ return;}
    if(tidy == 0 && tidx == 0){ printf("%i\n",row_ptrC[N*N]);}
    
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
                    double Aval = valA[l];
                    for( int j = row_ptrB[i]; j < row_ptrB[i+1]; ++j){
                        double entry = Aval*valB[j];
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
__global__ void addCRS(unsigned const int N, unsigned const int M, double* valA, int* colA, int* row_ptrA, double* valB, int* colB, int* row_ptrB, double* valC, int* colC, int* row_ptrC){

//    row_ptrC[0] = 0;
    extern __shared__ int offsets[];
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
//    printf("%i\n",tid);
    if(tid >= N) return;
    printf("%i\n",tid);
   // offsets[tid] = 0;

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
    for(unsigned int i = 0; i <limit; ++i){
        unsigned int tid = blockDim.y*i + threadIdx.y;
        unsigned int max = 0;
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
    if(threadIdx.y == 0) printf("%i\n", cs_o[(int)ceil((float)N/C)]);

}
__global__ void CRStoCELLCSIG(unsigned const int C, unsigned const int sigma, unsigned const int N, double* val, int* col, int* row_ptr, double* val_o,int* col_o, int* cs, int* cl){
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
                    col_o[cs[bid]+k*C+threadIdx.y] = 0;


                } else{

                    val_o[cs[bid]+k*C+threadIdx.y] = val[row_ptr[tid]+k];
                    col_o[cs[bid]+k*C+threadIdx.y] = col[row_ptr[tid]+k];
               }
        }

//        if(threadIdx.y == 0) printf("\t\tCl: %i, Cs: %i\n", cl[bid], cs[bid+1]);
    
    
    return;
}



__global__ void matvecCRS(unsigned const int N, double* valA, int* colA, int* row_ptrA, double* vec, double* res){
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    if(tid >= N) return;
    double tmp = 0;
    for(unsigned int i = row_ptrA[tid]; i < row_ptrA[tid+1]; ++i){
        tmp += valA[i]*vec[colA[i]];
    }
    res[tid] = tmp;
}

__global__ void matvecCELLCSIGMA(unsigned const int N, unsigned const int C, double* valCCS, int* colCCS, int* cs, int* cl, double* vec, double* res){
    int tid = blockDim.y*blockIdx.y + threadIdx.y;
    int bid = blockIdx.y;
    if(tid >= N) return;
    double tmp = 0;
    for(unsigned int i = 0; i < cl[bid]; ++i){
        tmp += valCCS[cs[bid]+i*C+threadIdx.y]*vec[colCCS[cs[bid]+i*C+threadIdx.y]];
    }
    res[tid] = tmp;
//    if(tid==0) printf("%i\n", cs[N]);
}


__global__ void lanczos(unsigned const int N, CELLCSIG_matrix* in, CELLCSIG_matrix* out){
    return;
}

void printCRS_matrix(unsigned const int N, double* val, int* col, int* row_ptr){
    printf("Row:\tCol:\tVal:\n");
    for(int i = 0; i < N; ++i){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; ++j){
            printf("(%i\t%i\t%lf)\n",row_ptr[i], col[j], val[j]);

        }
    
    }
    return;
}

void printCELLCSIG_matrix(unsigned const int N, unsigned const int C, double* valCCS, int* colCCS, int* cs, int* cl){
   
    for(int i = 0; i < (int)ceil((float)N/C); ++i){
        printf("Chunk start: %i, Chunk width: %i\n", cs[i], cl[i]);
        for(int l = 0; l < C; ++l){
            for(int j = 0; j < cl[i]; ++j){
                printf("[%i,\t%lf] ",colCCS[cs[i]+j*C+l], valCCS[cs[i]+j*C+l]);
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

