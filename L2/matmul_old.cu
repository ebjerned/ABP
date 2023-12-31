#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#define AssertCuda(error_code) \
if(error_code != cudaSuccess) 	\
{ \
	std::cout << "The cuda call in" << __FILE__ << " on line "\
	<< __LINE__ << " resulted in the error '" \
	<< cudaGetErrorString(error_code) << "'" << std::endl;\
	std::abort();\
}\

const int block_size = 128;
const int chunk_size = 1;


__global__ void set_vector(const int N, const float val, float *x)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        x[idx] = val;
    }
}

__global__ void set_vector_rising(const int N, const float val, float *x)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        x[idx] =val*(blockIdx.x*blockDim.x+threadIdx.x + threadIdx.y*gridDim.x*blockDim.x);
		
    }
}

__device__ void warpReduce0(volatile float* sdata, int tx){
    sdata[tx] += sdata[tx + 32];
    sdata[tx] += sdata[tx + 16];
    sdata[tx] += sdata[tx + 8];
    sdata[tx] += sdata[tx + 4];
    sdata[tx] += sdata[tx + 2];
    sdata[tx] += sdata[tx + 1];

}
template <unsigned int blockSize>
__device__ void warpReduce1(volatile float* sdata, int tx){
    if(blockSize >= 64) sdata[tx] += sdata[tx +32];
    if(blockSize >= 32) sdata[tx] += sdata[tx +16];
    if(blockSize >= 16) sdata[tx] += sdata[tx +8];
    if(blockSize >= 8) sdata[tx] += sdata[tx +4];
    if(blockSize >= 4) sdata[tx] += sdata[tx +2];
    if(blockSize >= 2) sdata[tx] += sdata[tx +1];


}

__global__ void matmat(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N, unsigned const K){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	
	if((col >= N)|| (row >= M)) return;


	int roof = (M+blockDim.y-1)/blockDim.y;
	roof = (row + blockDim.y*(roof-1)) < M ? roof : roof -1;
	for(unsigned int currentRow = 0; currentRow < N; ++currentRow){
		float sum = 0.f;
		for(unsigned int k= 0; k < roof; k++){
			float coeff = B[K*currentRow + col+ blockDim.y*k];
			sum += A[col*M+row + (blockDim.y*k)]*coeff;
	}
	
	__syncthreads();
	atomicAdd(&C[col+currentRow*K], sum);
   }
}

__global__ void matvec_old(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	
	if((col >= N)|| (row >= M)) return;

	int roof = (M+blockDim.y-1)/blockDim.y;
	roof = (row + blockDim.y*(roof-1)) < M ? roof : roof -1;
	float sum = 0.f;
	
	float coeff = B[col];
	for(unsigned int k= 0; k < roof; k++){
		sum = A[col*M+row + (blockDim.y*k)]*coeff;    
	    //__syncthreads();
    	//atomicAdd(&C[row+blockDim.y*k], sum);
        C[row+blockDim.y*k] += sum;
	}
	

   
}

__global__ void matvec0(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
	if((col >= N)|| (row >= M)){
        sdata[tx]=0;
    }else{
        sdata[tx] = A[col*M+row]*B[col];
    }

    __syncthreads();
    for(unsigned int s = 1; s<blockDim.x; s*=2){
        if(tx %(2*s) == 0){
            sdata[tx]+=sdata[tx+s];
        }
        __syncthreads();
    
    }

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

   
}

__global__ void matvec1(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
	if((col >= N)|| (row >= M)){
        sdata[tx]=0;
    }else{
        sdata[tx] = A[col*M+row]*B[col];
    }

    __syncthreads();
    for(unsigned int s = 1; s<blockDim.x; s*=2){
        int index = 2*s*tx;
        if(index < blockDim.x){
            sdata[index]+=sdata[index+s];
        }
        __syncthreads();
    
    }

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

   
}

__global__ void matvec2(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
	if((col >= N)|| (row >= M)){
        sdata[tx]=0;
    }else{
        sdata[tx] = A[col*M+row]*B[col];
    }

    __syncthreads();
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tx < s){
            sdata[tx]+=sdata[tx+s];
        }
        __syncthreads();
    
    }

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

   
}
__global__ void matvec3(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = (blockDim.x*2)*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;

    float sum = ((col >= N) || (row >= M)) ? 0 : A[col*M+row]*B[col];
    if((col + blockDim.x) < N){
        sum += A[(col+blockDim.x)*M+row]*B[col+blockDim.x];
    }
    sdata[tx] = sum;

/*
    if((col >= N)|| (row >= M) || ((col+blockDim.x) >= N)){
        sdata[tx]=0;
        //if((col+blockDim.x) >= N) printf("\t %i\n", col + blockDim.x);
    }else{
        sdata[tx] = A[col*M+row]*B[col] + A[(col+blockDim.x)*M+row]*B[col+blockDim.x];

    }
*/
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tx < s){
            sdata[tx]+=sdata[tx+s];
        }
        __syncthreads();
    
    }

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

}   
__global__ void matvec4(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = (blockDim.x*2)*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;

    float sum = ((col >=N) || (row >=M)) ? 0 :A[col*M+row]*B[col];
    if((col + blockDim.x) < N) sum += A[(col+blockDim.x)*M+row]*B[col+blockDim.x];
    sdata[tx] = sum;


    __syncthreads();
    for(unsigned int s = blockDim.x/2; s>32; s>>=1){
        if(tx < s){
            sdata[tx]+=sdata[tx+s];
        }
        __syncthreads();
    
    }
    if(tx < 32) warpReduce0(sdata, tx);

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

}
template <unsigned int blockSize>
__global__ void matvec(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = (blockDim.x*2)*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;

    float sum = ((col >=N) || (row >=M)) ? 0 :A[col*M+row]*B[col];
    if((col + blockDim.x) < N) sum += A[(col+blockDim.x)*M+row]*B[col+blockDim.x];
    sdata[tx] = sum;


    __syncthreads();
/*    for(unsigned int s = blockDim.x/2; s>32; s>>=1){
        if(tx < s){
            sdata[tx]+=sdata[tx+s];
        }
        __syncthreads();
    
    }*/
    if(blockSize >= 1024){
        if(tx < 512) {sdata[tx] += sdata[tx+512];} __syncthreads();}
    if(blockSize >= 512){
        if(tx < 256) {sdata[tx] += sdata[tx+256];} __syncthreads();}
    if(blockSize >= 256){
        if(tx < 128) {sdata[tx] += sdata[tx+128];} __syncthreads();}
    if(blockSize >= 128){
        if(tx < 64) {sdata[tx] += sdata[tx+64];} __syncthreads();}
    if(tx < 32) warpReduce1<blockSize>(sdata, tx);

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

}

template <unsigned int blockSize>
__global__ void matvec6(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
    __shared__ float sdata[block_size];

    int col = (blockDim.x*2)*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
    int grid = blockSize*2*gridDim.x;
//    float sum = ((col >=N) || (row >=M)) ? 0 :A[col*M+row]*B[col];
//    if((col + blockDim.x) < N) sum += A[(col+blockDim.x)*M+row]*B[col+blockDim.x];
//    sdata[tx] = sum;
    sdata[tx] = 0;  
    while (col < N){
        if(row < M) sdata[tx] += A[col*M+row]*B[col];
        if((col+blockDim.x)<N) sdata[tx] += A[(col+blockDim.x)*M+row]*B[col+blockDim.x];
        col += grid;
    }

    __syncthreads();

    if(blockSize >= 1024){
        if(tx < 512) {sdata[tx] += sdata[tx+512];} __syncthreads();}
    if(blockSize >= 512){
        if(tx < 256) {sdata[tx] += sdata[tx+256];} __syncthreads();}
    if(blockSize >= 256){
        if(tx < 128) {sdata[tx] += sdata[tx+128];} __syncthreads();}
    if(blockSize >= 128){
        if(tx < 64) {sdata[tx] += sdata[tx+64];} __syncthreads();}
    if(tx < 32) warpReduce1<blockSize>(sdata, tx);

    if(tx == 0) atomicAdd(&C[row], sdata[0]);

}
__global__ void matvecT(const float* A, const float* B, float* C, unsigned const int M , unsigned const int N	){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	
	if((col >= N)|| (row >= M)) return;

	int roof = (M+blockDim.y-1)/blockDim.y;
	roof = (row + blockDim.y*(roof-1)) < M ? roof : roof -1;
	float sum = 0.f;
	float coeff = B[col];
	for(unsigned int k= 0; k < roof; k++){
		sum += A[col+row*N + (blockDim.y*k)]*coeff;
	}
	
	__syncthreads();
	atomicAdd(&C[col], sum);
   
}
void matmat_naive(const float* A, const float* B, float* C, unsigned const int M, unsigned const int N, unsigned const K){
	for(int i = 0; i < M;++i)
		for(int k = 0; k < K; ++k){
			C[i+M*k]=0.f;
			for(int j = 0; j < N; ++j)
				C[i+M*k] += A[i+j*M]*B[j+k*N];
		}

}

void matmat_naiveT(const float* A, const float* B, float* C, unsigned const int M, unsigned const int N, unsigned const K){
	for(int i = 0; i < M;++i)
		for(int k = 0; k < K; ++k){
			C[i+M*k]=0.f;
			for(int j = 0; j < N; ++j)
				C[i+M*k] += A[i*N+j]*B[j+k*N];
		}

}
void benchmark_mat(  const std::size_t M,
					 const std::size_t N,
                     const std::size_t K)
{

  cudaError_t errorCode;

  float *A, *B, *C;
  errorCode = cudaMalloc(&A, M * N * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&B, N * K * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&C, M * K* sizeof(float));
  AssertCuda(errorCode);


  set_vector<<<(M*N+block_size-1)/block_size, block_size>>>(M*N, 1.f/*/sqrt(N)*/, A);
 
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<(N*K+block_size-1)/block_size, block_size>>>(N*K, 1.f/*/sqrt(N)*/, B);
 
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<(M*K+block_size-1)/block_size, block_size>>>(M*K, 0.f, C);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);

  std::vector<float> result_host(M*K);
  
  dim3 gridDim(ceil(0.5f*(float)M/(float)block_size),N);
  dim3 blockDim(block_size,1);
  
  const unsigned int n_tests = 20;
  const unsigned int n_repeat = 20;
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {

 	  set_vector<<<(M*K+block_size-1)/block_size, block_size>>>(M*K, 0.f, C);
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep){
 	 //   set_vector<<<(M*K+block_size-1)/block_size, block_size>>>(M*K, 0.f, C);

//		if(K > 1){
//			matmat<<<gridDim, blockDim>>>(A, B, C, M, N, K);
//		}else{
            matvec<block_size><<<gridDim, blockDim>>>(A, B, C, M, N);
//		}
//	    errorCode = cudaGetLastError();
//	    AssertCuda(errorCode);
	  }
      cudaDeviceSynchronize();

	//	matmat_naive(A, B, C, M, N, K);

      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  errorCode = cudaMemcpy(result_host.data(),  C, M *K* sizeof(float), cudaMemcpyDeviceToHost);  
  AssertCuda(errorCode);

 //Printing for checking correctness
/* for(unsigned int i = 0; i <M*K;++i){
  	std::cout << result_host[(i*M)%(M*K)+(i/K)] << " ";
	if (i % K == K-1) std::cout << "" << std::endl;
  }*/
/*  
  for(unsigned int i = 0; i < M; ++i)
  	std::cout << result_host[i] << std::endl;
*/
  errorCode = cudaFree(A);
  AssertCuda(errorCode);
  errorCode = cudaFree(B);
  AssertCuda(errorCode);
  errorCode = cudaFree(C);
  AssertCuda(errorCode);

  std::cout << "MATMUL (GPU) of size (M,N,K) " << std::setw(8) << M << "  " << N << " " << K 
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-9 * 2 * N * M * N / best
            << " GFLOPS/s or " << std::setw(8)
            << 1e-9 * sizeof(float) *(N*M + M + N) / best << " GB/s" << std::endl; 

}

int main(int argc, char **argv)
{
  if (argc % 2 == 0)
    {
      std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << "-M rows -N columns/rows -K columns" << std::endl;
      std::abort();
    }

  long M  = 8;
  long N  = -1;
  long K = 1;
  // parse from the command line
  for (int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-M")
        M = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-N")
        N = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-K")
        K = static_cast<long>(std::stod(argv[l + 1]));
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }
  if(N < 0) N = M;

  //For running series test
for(float i = 7; i < 14; i+= 0.2){
  		long size = round(pow(2,i));
		benchmark_mat(size,size,K);
  }


// benchmark_mat(M, N, K);

  return 0;
}
