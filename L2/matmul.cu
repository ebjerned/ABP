
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




const int block_size = 64;
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
//	__syncthreads();
//	atomicAdd(&b[index],sum);
}



// Run the actual benchmark
void benchmark_triad(const bool        align,
                     const std::size_t N,
                     const long long   repeat)
{

  unsigned int elementsSidePerBlock = 1;
  dim3 blockDimensions(ceil(N/elementsSidePerBlock),ceil(N/elementsSidePerBlock));
  cudaError_t errorCode;

  float *v1, *v2, *v3;
  // allocate memory on the device
  errorCode = cudaMalloc(&v1, N * N * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&v2, N * sizeof(float));
  AssertCuda(errorCode);
  errorCode = cudaMalloc(&v3, N * sizeof(float));
  AssertCuda(errorCode);

  const unsigned int n_blocks = (N + block_size - 1) / block_size;

  set_vector<<<blockDimensions, block_size>>>(N*N, 1.f, v1);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<blockDimensions, block_size>>>(N, 1.f, v2);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);
  set_vector<<<blockDimensions, block_size>>>(N, 0.f, v3);
  errorCode = cudaGetLastError();
  AssertCuda(errorCode);

  std::vector<float> result_host(N);
  dim3 gridDim(ceil(N/block_size));
  dim3 blockDim(block_size);
  const unsigned int           n_tests = 20;
  /*const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);*/
  const unsigned int n_repeat = 1;
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep)
      //matmul<<<blockDimensions, block_size>>>(v1, v2, v3, N, N);
      matvec<<<gridDim, blockDim>>>(v1, v2, v3, N, N);
	  errorCode = cudaGetLastError();
  	  AssertCuda(errorCode);
      cudaDeviceSynchronize();
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
  errorCode = cudaMemcpy(result_host.data(), v1, N*N * sizeof(float), cudaMemcpyDeviceToHost);  
  AssertCuda(errorCode);


 for(unsigned int i = 0; i <N*N;++i){
  	std::cout << result_host[(i*N)%(N*N)+(i/N)] << " ";
	if (i % N == N-1) std::cout << "" << std::endl;
  }
  
/*  for(unsigned int i = 0; i < N; ++i)
  	std::cout << result_host[i] << std::endl;*/
  //Not perfect check for correctness, works for 8 but not for 512 or larger
  if (result_host[0] != N*((N-1)*N*(2*N-1)/6))
    std::cout << "Error in computation, got "
              << (result_host[0] + result_host[N - 1]) << " instead of 526"
              << std::endl;

  // Free the memory on the device
  errorCode = cudaFree(v1);
  AssertCuda(errorCode);
  errorCode = cudaFree(v2);
  AssertCuda(errorCode);
  errorCode = cudaFree(v3);
  AssertCuda(errorCode);

  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-9 * 2 * N * N * N / best
            << " GFLOPS/s or " << std::setw(8)
            << 1e-9 * 3 * sizeof(float) * N / best << " GB/s" << std::endl;
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

  long N_min  = 8;
  long N_max  = -1;
  bool align  = false;
  long repeat = -1;
  // parse from the command line
  for (int l = 1; l < argc; l += 2)
    {
      std::string option = argv[l];
      if (option == "-min")
        N_min = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-max")
        N_max = static_cast<long>(std::stod(argv[l + 1]));
      else if (option == "-repeat")
        repeat = std::atoll(argv[l + 1]);
      else if (option == "-align")
        align = std::atoi(argv[l + 1]);
      else
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }
  if (N_min < 1)
    {
      std::cout << "Expected positive size for min argument, got " << N_min
                << std::endl;
      return 0;
    }

  if (N_max < N_min)
    N_max = N_min;

  for (long n = N_min; n <= N_max; n = (1 + n * 1.1))
    {
      // round up to nearest multiple of 8
      n = (n + 7) / 8 * 8;
      benchmark_triad(align, n, repeat);
    }

  return 0;
}
