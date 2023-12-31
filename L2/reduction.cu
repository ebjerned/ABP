
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#define AssertCuda(error_code) 									\
if(error_code != cudaSuccess) 									\
{ 																\
	std::cout << "The cuda call in " << __FILE__ << " on line " \
	<< __LINE__ << " resulted in the error '" 					\
	<< cudaGetErrorString(error_code) << "'" << std::endl; 		\
	std::abort();												\
}														 		\


const int block_size =64;
const int chunk_size = 1;


__device__ void warpReduce0(volatile float* sdata, int tid){
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__device__ void warpReduce1(volatile float* sdata, int tid){
	if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void reduce0(float* g_idata, float* result){
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	__syncthreads();

	for(unsigned int s=1; s<blockDim.x; s*=2){
		if(tid%(2*s)==0){
			sdata[tid] += sdata[tid+s];	
		}
		__syncthreads();
	}
	if(tid==0)
		atomicAdd(result, sdata[0]);	
}

__global__ void reduce1(float* g_idata, float* result){
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	__syncthreads();

	for(unsigned int s=1; s<blockDim.x; s*=2){
		int index = 2*s*tid;
		if(index < blockDim.x){
			sdata[index] += sdata[index+s];	
		}
		__syncthreads();
	}
	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}


__global__ void reduce2(float* g_idata, float* result){
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>0; s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid+s];
		
		}
		__syncthreads();
	}
	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}


__global__ void reduce3(float* g_idata, float* result){
    // When calling, start kernel with half n_blocks
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
	sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>0; s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid+s];
		
		}
		__syncthreads();
	}
	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}

__global__ void reduce4(float* g_idata, float* result){
    // When calling, start kernel with half n_blocks
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
	sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>32; s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid+s];
			__syncthreads();
		}

	}
	if (tid < 32) warpReduce0(sdata, tid);

	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}

template <unsigned int blockSize>
__global__ void reduce5(float* g_idata, float* result){
    // When calling, start kernel with half n_blocks
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
	sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
	__syncthreads();

	if(blockSize >= 2048){
		if(tid<1024){sdata[tid]+=sdata[tid+1024];}__syncthreads();}
	if(blockSize >= 1024){
		if(tid<512){sdata[tid]+=sdata[tid+512];}__syncthreads();}
	if(blockSize >= 512){
		if(tid<256){sdata[tid]+=sdata[tid+256];}__syncthreads();}
	if(blockSize >= 256){
		if(tid<128){sdata[tid]+=sdata[tid+128];}__syncthreads();}
	if(blockSize >= 128){
		if(tid<64){sdata[tid]+=sdata[tid+64];}__syncthreads();}

	if (tid < 32) warpReduce1<blockSize>(sdata, tid);

	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}

template <unsigned int blockSize>
__global__ void reduce6(float* g_idata, float* result){
    // When calling, start kernel with half n_blocks
	__shared__ float sdata[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while(i<4194304){
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i+=gridSize;	
	}
	__syncthreads();

	if(blockSize >= 2048){
		if(tid<1024){sdata[tid]+=sdata[tid+1024];}__syncthreads();}
	if(blockSize >= 1024){
		if(tid<512){sdata[tid]+=sdata[tid+512];}__syncthreads();}
	if(blockSize >= 512){
		if(tid<256){sdata[tid]+=sdata[tid+256];}__syncthreads();}
	if(blockSize >= 256){
		if(tid<128){sdata[tid]+=sdata[tid+128];}__syncthreads();}
	if(blockSize >= 128){
		if(tid<64){sdata[tid]+=sdata[tid+64];}__syncthreads();}

	if (tid < 32) warpReduce1<blockSize>(sdata, tid);

	
	if(tid==0){
		atomicAdd(result, sdata[0]);	

	}
}



__global__ void set_vector(const int N, const float val, float *x)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        x[idx] = 1;
    }
}


// Run the actual benchmark
void benchmark_triad(const bool        align,
                     const std::size_t N,
                     const long long   repeat)
{
  float *v1;
  float* result;
  float res = 0;
  cudaError_t error_code;
  // allocate memory on the device
  error_code = cudaMalloc((void**) &v1, N * sizeof(float));
  AssertCuda(error_code);
  //error_code = cudaMalloc(&v2, N * sizeof(int));
  //AssertCuda(error_code);
  error_code = cudaMalloc((void**) &result, sizeof(float));
  AssertCuda(error_code);
  const unsigned int n_blocks = (N + block_size - 1) / block_size;

  set_vector<<<n_blocks, block_size>>>(N, 17, v1);
  error_code = cudaGetLastError();
  AssertCuda(error_code);
 /* 
  set_vector<<<n_blocks, block_size>>>(N, 0, v2);
  error_code = cudaGetLastError();
  AssertCuda(error_code);
  */

  std::vector<float> result_host(N);

  const unsigned            n_tests = 20;
  const unsigned long long int n_repeat =
    repeat > 0 ? repeat : std::max(1UL, 100000000U / N);
  double best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      for (unsigned int rep = 0; rep < n_repeat; ++rep){
	    
		error_code = cudaMemset(result,0,sizeof(float));
		AssertCuda(error_code);
		cudaDeviceSynchronize();
        reduce6<block_size><<<n_blocks/2, block_size>>>(v1, result);
  		error_code = cudaGetLastError();
  		AssertCuda(error_code);
		

	  }

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
  error_code = cudaMemcpy(result_host.data(), v1, N*sizeof(float), cudaMemcpyDeviceToHost);
  AssertCuda(error_code);
  error_code = cudaMemcpy(&res, result, sizeof(float), cudaMemcpyDeviceToHost);
  AssertCuda(error_code);
  std::cout << "Sum: " << res << std::endl;
  if (res != N)
    std::cout << "Error in computation, got "
              << (result_host[0] )<< " instead of 526"
              << std::endl;
  /*for(unsigned int i = 0; i < N; i++){
  	std::cout << result_host[i] << " ";
  }*/
  std::cout << "Finished printout" << std::endl;
  // Free the memory on the device
  cudaFree(v1);
  //cudaFree(v2);
  //cudaFree(v3);
  cudaFree(result);
  std::cout << "STREAM triad of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8)
            << 1e-9 * sizeof(float) * N / best << " GB/s" << std::endl;
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
//      n = (n + 7) / 8 * 8;
	  n = pow(2, ceil(log(n)/log(2)));
      benchmark_triad(align, n, repeat);
    }

  return 0;
}

