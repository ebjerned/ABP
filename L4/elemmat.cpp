#include <Kokkos_Core.hpp>
#include <cstdlib>


int main(int argc, char* argv[]){
    std::cout << "Hello world" << std::endl;

    unsigned int N;
    unsigned int N_max;

    if (argc == 3){
        if( strcmp(argv[1], "-Nele") == 0){
            N = atoi(argv[2]);
            N_max = N;
        }
    }else if (argc == 5){
        if( strcmp(argv[1], "-Nele") == 0){
            N = atoi(argv[2]);
        }
        if(strcmp(argv[3], "-Nmax") == 0){
            N_max = atoi(argv[4]);
        }
    }else {
        std::cout << "Usage: ./elemmat -Nele Number of element" << std::endl;
    }

    
        Kokkos::initialize(argc, argv);
        {
        for(unsigned int i = N; i <= N_max; i += (N_max-N)/20){
            unsigned int N_curr = i;
            #ifdef KOKKOS_ENABLE_CUDA
            #define MemSpaceGPU Kokkos::CudaSpace
            #define MemSpaceCPU Kokkos::HostSpace
            #endif
            #ifdef KOKKOS_ENABLE_OPENMPTARGET
                #ifdef KOKKOS_ENABLE_CUDA
                #define MemSpaceGPU Kokkos::CudaSpace
                #else
                #define MemSpaceGPU Kokkos::OPENMPTargetSpace
                #endif
                #define MemSpaceCPU Kokkos::OpenMPTargetSpace
            #endif
    
            #ifndef MemSpaceCPU
            #define MemSpaceCPU Kokkos::HostSpace
            #endif

            #ifndef MemSpaceGPU
            #define MemSpaceGPU Kokkos::HostSpace
            #endif
            #define Layout Kokkos::LayoutRight

            using ExecSpaceGPU = MemSpaceGPU::execution_space; 
            using ExecSpaceCPU = MemSpaceCPU::execution_space;
            typedef double precision_t ;
            typedef Kokkos::View<precision_t*[3][3], Layout, MemSpaceGPU> ViewJType;
            typedef Kokkos::View<precision_t*[4][4], Layout, MemSpaceGPU> ViewAType;
    
            ViewJType J("J", N_curr);
            ViewAType A("A", N_curr);

            ViewJType::HostMirror h_J = Kokkos::create_mirror_view(J);
            ViewAType::HostMirror h_A = Kokkos::create_mirror_view(A);
/*
            for(unsigned int i = 0; i < N_curr; ++i){
                h_J(i,0,0) = 3;
                h_J(i,0,1) = 1;
                h_J(i,0,2) = 1;
                h_J(i,1,0) = 1;
                h_J(i,1,1) = 3;
                h_J(i,1,2) = 1;
                h_J(i,2,0) = 1;
                h_J(i,2,1) = 1;
                h_J(i,2,2) = 3;
            }

            for(unsigned int i = 0; i < N_curr; ++i){
                A(i,0,0) = 0;
                A(i,0,1) = 0;
                A(i,0,2) = 0;
                A(i,0,3) = 0;
                A(i,1,0) = 0;
                A(i,1,1) = 0;
                A(i,1,2) = 0;
                A(i,1,3) = 0;
                A(i,2,0) = 0;   
                A(i,2,1) = 0;
                A(i,2,2) = 0;
                A(i,2,3) = 0;
                A(i,3,0) = 0;
                A(i,3,1) = 0;
                A(i,3,2) = 0;
                A(i,3,3) = 0;

            }


*/

        Kokkos::parallel_for("J-assembler",Kokkos::MDRangePolicy<ExecSpaceCPU, Kokkos::Rank<3>>({0,0,0},{N,3,3}), [=] (const int i, const int j, const int k) {
            h_J(i,j,k) = 1;
        });

        Kokkos::parallel_for("J-assembler2",Kokkos::MDRangePolicy<ExecSpaceCPU, Kokkos::Rank<2>>({0,0},{N,3}), [=] (const int i, const int j) {
            h_J(i,j,j) = 3;
        });


        Kokkos::parallel_for("A-init", Kokkos::MDRangePolicy<ExecSpaceCPU, Kokkos::Rank<3>>({0,0,0},{N,4,4}),[=] (const int i, const int j, const int k){
            h_A(i,j,k) = 0;
        });

        Kokkos::deep_copy(J, h_J);
        Kokkos::deep_copy(A, h_A);
        Kokkos::fence();
        Kokkos::Timer timer;

        Kokkos::parallel_for("A-assembler", Kokkos::RangePolicy<ExecSpaceGPU>(0,N_curr), KOKKOS_LAMBDA (const int i){
            precision_t C0 = J(i,1,1)*J(i,2,2)-J(i,1,2)*J(i,2,1); // 3 flops
            precision_t C1 = J(i,1,2)*J(i,2,0)-J(i,1,0)*J(i,2,2); // 3 flops
            precision_t C2 = J(i,1,0)*J(i,2,1)-J(i,1,1)*J(i,2,0); // 3 flops
            precision_t inv_J_det = J(i,0,0) * C0 + J(i, 0, 1)*C1 + J(i,0,2) *C2; // 5 flops
            precision_t d = (1./6.)/inv_J_det; // 2 flops
            precision_t G0 = d*(J(i,0,0)*J(i,0,0)+J(i,1,0)*J(i,1,0)+J(i,2,0)*J(i,2,0)); // 6 flops v
            precision_t G1 = d*(J(i,0,0)*J(i,0,1)+J(i,1,0)*J(i,1,1)+J(i,2,0)*J(i,2,1));
            precision_t G2 = d*(J(i,0,0)*J(i,0,2)+J(i,1,0)*J(i,1,2)+J(i,2,0)*J(i,2,2));
            precision_t G3 = d*(J(i,0,1)*J(i,0,1)+J(i,1,1)*J(i,1,1)+J(i,2,1)*J(i,2,1));
            precision_t G4 = d*(J(i,0,1)*J(i,0,2)+J(i,1,1)*J(i,1,2)+J(i,2,1)*J(i,2,2));
            precision_t G5 = d*(J(i,0,2)*J(i,0,2)+J(i,1,2)*J(i,1,2)+J(i,2,2)*J(i,2,2));//          ^
        
            A(i,0,0) = G0;
            A(i,0,1) = A(i,1,0) = G1;
            A(i,0,2) = A(i,2,0) = G2;
            A(i,0,3) = A(i,3,0) = -G0 -G1 -G2; //3 flops
            A(i,1,1) = G3;
            A(i,1,2) = A(i,2,1) = G4;
            A(i,1,3) = A(i,3,1) = -G1 -G3 -G4; //3 flops
            A(i,2,2) = G5;
            A(i,2,3) = A(i,3,2) = -G2 -G4 -G5; //3 flops
            A(i,3,3) = G0 + 2*G1 + 2*G2 + G3 + 2*G4 + G5; // 8 flops
            // 63 flops per element
        });
        Kokkos::fence();
        double time = timer.seconds();

        Kokkos::deep_copy(h_A, A);
        unsigned int Melements_per_second = N_curr*1e-6/time;
        float GFLOPS_per_second = 63*N_curr*1e-9/time;
        float GB_per_second = sizeof(precision_t)*(16*N_curr+9*N_curr)*1e-9/time;

        std::cout << "Element calculation test with N = " << N_curr << " Time: " << time << " Melement/s: " << Melements_per_second << " GFLOPS/s: " << GFLOPS_per_second << " GB/s: " << GB_per_second << std::endl;
/*      for(unsigned int i = 0; i < 16; ++i){
           std::cout << A(0, i%4, i/4) << "\t";
           if(i%4 == 3) std::cout << "" << std::endl;
        }*/

        }
    }
    Kokkos::finalize();

    return 0;


}
