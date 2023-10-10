/* ---------------------------------------------
Makefile constructed configuration:
----------------------------------------------*/
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

#define KOKKOS_VERSION 40100
#define KOKKOS_VERSION_MAJOR 4
#define KOKKOS_VERSION_MINOR 1
#define KOKKOS_VERSION_PATCH 00

/* Execution Spaces */
#define KOKKOS_ENABLE_OPENMP
/* General Settings */
#define KOKKOS_ENABLE_DEPRECATED_CODE_4
#define KOKKOS_ENABLE_CXX17
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_LIBDL
/* Optimization Settings */
/* Cuda Settings */
#define KOKKOS_ARCH_AVX
#define KOKKOS_ENABLE_IMPL_MDSPAN
