
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

__global__ void compute_triad(const int    N,
                              const float  a,
                              const float *x,
                              const float *y,
                              float *      z)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        z[idx] = a * x[idx] + y[idx];
    }
}


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
        x[idx] = val*(blockIdx.x*blockDim.x+threadIdx.x + threadIdx.y*gridDim.x*blockDim.x);
    }
}



__global__ void matmul(const float*A, const float*B, float* C, unsigned const int M, unsigned const int N){
	size_t i = blockIdx.y*blockDim.y + threadIdx.y;
	size_t j = blockIdx.x*blockDim.x + threadIdx.x;

	if((i>= M) || (j >= N)){
		return;
	}

	float acc_sum = 0;
	for(unsigned int k = 0; k<N; ++k)
	{
		acc_sum += A[i+k*N]*B[k+j*N];
	}
//	__syncthreads();
	
	C[i*N+j] = acc_sum;

}

__global__ void matvec(const float* A, const float* x, float* b, unsigned const int M, unsigned const int N){
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	float sum = 0.f;
	for(unsigned int k = 0; k < N; ++k)
		sum += A[N*index + k]*x[k];

	b[index] = sum;
}

__global__ void matvec2(const float* A, const float* x, float*b, unsigned const int M, unsigned const int N){
	int col = blockDim.x*blockIdx.x +threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	float sum = 0.f;
	b[row] = 0; //TODO: DO this outside of kernel
	if((col >= N)|| (row >= M)) return;
	int roof = (N+blockDim.x-1)/blockDim.x;
	for(unsigned int k = 0; k < roof; ++k){
			if((col + blockDim.x*k) >= N) break; // TODO: This can be moved to when calculating roof
			sum += A[row*N+col+(blockDim.x*k)]*x[col+(blockDim.x*k)];
	}
	__syncthreads();
	atomicAdd(&b[row],sum);
}

__global__ void matmat(const float* A, const float* B, float* C, unsigned const int M, unsigned const int N, unsigned const K){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	float sum = 0.f;
	if((col >= N)|| (row >= M)) return;
	int roof = (N+blockDim.x-1)/blockDim.x;
	roof = (col + blockDim.x*(roof-1)) < N ? roof : roof -1;
	for(unsigned int currentCol = 0; currentCol < K; ++currentCol){
		sum = 0.f;
		for(unsigned int k = 0; k < roof; ++k){
			sum += A[col*M+row+(blockDim.x*k)]*B[currentCol*N + col+(blockDim.x*k)];
		}
		__syncthreads();
		atomicAdd(&C[row+currentCol*M], sum);

	}
}

__global__ void matmatT(const float* A, const float* B, float* C, unsigned const int M, unsigned const int N, unsigned const K){
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	float sum = 0.f;
	if((col >= N)|| (row >= M)) return;
	int roof = (N+blockDim.x-1)/blockDim.x;
	roof = (col + blockDim.x*(roof-1)) < N ? roof : roof -1;
	for(unsigned int currentCol = 0; currentCol < K; ++currentCol){
		sum = 0.f;
		for(unsigned int k = 0; k < roof; ++k){
			sum += A[col+row*N+(blockDim.x)*N]*B[currentCol*N + col+(blockDim.x*k)];
		}
		__syncthreads();
		atomicAdd(&C[row+currentCol*M], sum);

	}
}
void matmat_naive(const float* A, const float* B, float* C, unsigned const int M, unsigned const int N, unsigned const K){
	for(int i = 0; i < M;++i)
		for(int k = 0; k < K; ++k){
			C[i+M*k]=0.f;
			for(int j = 0; j < N; ++j)
				C[i+M*k] += A[i+j*M]*B[j+k*N];
		}

}

// Run the actual benchmark
void benchmark_mat(  const std::size_t M,
					 const std::size_t N,
                     const std::size_t K)
{

  unsigned int elementsSidePerBlock = 1;
  dim3 blockDimensions(ceil(N/elementsSidePerBlock),ceil(N/elementsSidePerBlock));
  cudaError_t errorCode;

  float *A, *B, *C;
  // allocate memory on the device
/*  errorCode = cudaMalloc(&A, M * N * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&B, N * K * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&C, M * K* sizeof(float));
  AssertCuda(errorCode);

  const unsigned int n_blocks = (M*N + block_size - 1) / block_size;

  set_vector<<<n_blocks, block_size>>>(M*N, 1.f/sqrt(N), A);
 
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<(N*K+block_size-1)/block_size, block_size>>>(N*K, 1.f/sqrt(N), B);
  //dim3 risingDimB(ceil(N/elementsSidePerBlock), ceil(K/elementsSidePerBlock));
  //set_vector_rising<<<risingDimB, block_size>>>(N*K, 1.f, B);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<blockDimensions, block_size>>>(M*K, 0.f, C);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
*/

	A = (float*)malloc(M*N*sizeof(float));
	B = (float*)malloc(N*K*sizeof(float));
	C = (float*)malloc(M*K*sizeof(float));
	memset(A, 1.f, M*N*sizeof(float));
	memset(B, 1.f, N*K*sizeof(float));
	memset(C, 0, M*K*sizeof(float));
/*	for(unsigned int i = 0; i < M*K; i++){
		A[i] = 1.f;
		B[i] = 1.f;
		C[i] = 0.f;
	}*/

  std::vector<float> result_host(M*K);
  dim3 gridDim(1,M);
  dim3 blockDim(block_size,1);
  const unsigned int           n_tests = 20;
  /*const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);*/
  const unsigned int n_repeat = 1;
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();
/*
      for (unsigned int rep = 0; rep < n_repeat; ++rep){
  		set_vector<<<(M*K+block_size-1)/block_size, block_size>>>(M*K, 0.f, C);
		matmat<<<gridDim, blockDim>>>(A, B, C, M, N, K);
	    errorCode = cudaGetLastError();
  	    AssertCuda(errorCode);
	  }
      cudaDeviceSynchronize();*/

		matmat_naive(A, B, C, M, N, K);

      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  // Copy the result back to the host
  //errorCode = cudaMemcpy(result_host.data(), C , M *K* sizeof(float), cudaMemcpyDeviceToHost);  
  //AssertCuda(errorCode);

/*
 for(unsigned int i = 0; i <M*K;++i){
  	std::cout << result_host[(i*M)%(M*K)+(i/K)] << " ";
	if (i % K == K-1) std::cout << "" << std::endl;
  }
  
  for(unsigned int i = 0; i < M; ++i)
  	std::cout << result_host[i] << std::endl;

  */
  //Not perfect check for correctness, works for 8 but not for 512 or larger
  //if (result_host[0] != N*((N-1)*N*(2*N-1)/6))
/*    std::cout << "Computation got "
              << (result_host[0]) << " out of " << 1
              << std::endl;
*/
  // Free the memory on the device
/*  errorCode = cudaFree(A);
  AssertCuda(errorCode);
  errorCode = cudaFree(B);
  AssertCuda(errorCode);
  errorCode = cudaFree(C);
  AssertCuda(errorCode);*/

  free(A);
  free(B);
  free(C);
//  if( result_host[0]  < (1+std::numeric_limits<float>::epsilon()) && result_host[0] > (1 - std::numeric_limits<float>::epsilon())){
  std::cout << "STREAM triad of size " << std::setw(8) << M << "  " << N << " " << K 
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-9 * 2 * N * M * N / best
            << " GFLOPS/s or " << std::setw(8)
            << 1e-9 * sizeof(float) *(N*M + M + N) / best << " GB/s" << std::endl; 

//  } else {
//	std::cout << "Invalid: Error to large at dim" << M << " " << N << " " << K << std::endl;

  //}
}

int main(int argc, char **argv)
{
  if (argc % 2 == 0)
    {
      std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << "-min 100 -max 1e8 -repeat -1" << std::endl;
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
  for(float i = 10.6; i < 12.4; i+= 0.2){
  	long size = round(pow(2,i));
	benchmark_mat(size,size,size);
  }
 // benchmark_mat(M, N, K);

  return 0;
}
