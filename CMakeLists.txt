cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PARALLEL_MESH LANGUAGES CXX CUDA)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-ccbin='C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019\\windows\\bin\\intel64\\icl.exe'")
find_package(CUDA REQUIRED)
#set(CMAKE_CUDA_COMPILER "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe")
#set(CUDA_HOST_COMPILER "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2019\windows\bin\intel64\icl.exe")


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
link_directories(${PROJECT_NAME} $ENV{TBB_ROOT}\\lib\\intel64_win\\vc14)
#target_link_libraries(${PROJECT_NAME} $ENV{TBB_ROOT}\\lib\\intel64_win\\vc14\\tbb.lib)

file(GLOB SOURCES
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.cuh"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        )


add_executable(${PROJECT_NAME} ${SOURCES}) 

#set(CMAKE_GENERATOR_TOOLSET "Intel C++ Compiler 19.0" CACHE STRING "Platform Toolset" FORCE)

if(MSVC)
    #parallel stl activate
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$ENV{PSTL_ROOT}\\include;$ENV{TBB_ROOT}\\include;$(IncludePath)")

    #works with intel compiler 19
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")

else(MSVC)
    message(FATAL_ERROR "\nError: Only Microsoft Visual Studio is currently supported.\n")
endif(MSVC)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
