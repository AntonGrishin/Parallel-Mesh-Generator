cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PARALLEL_MESH LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)

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

if( CMAKE_VERSION VERSION_LESS "3.1" )
    if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
else( CMAKE_VERSION VERSION_LESS "3.1" )
    set( CMAKE_CXX_STANDARD 11 )
endif( CMAKE_VERSION VERSION_LESS "3.1" )
add_executable(${PROJECT_NAME} ${SOURCES})

if(MSVC)
    #parallel stl activate
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$ENV{PSTL_ROOT}\\include;$ENV{TBB_ROOT}\\include;$(IncludePath)")

    #works with intel compiler 19
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qopenmp-simd /Qtbb")

else(MSVC)
    message(FATAL_ERROR "\nError: Only Microsoft Visual Studio is currently supported.\n")
endif(MSVC)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
