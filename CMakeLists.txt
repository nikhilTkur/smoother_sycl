cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME multigrid)
project(${PROJECT_NAME})


# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

set(CMAKE_CXX_FLAGS "-Ofast -fsycl -fsycl-unnamed-lambda -DMKL_ILP64 -tbb ${CMAKE_CXX_FLAGS} -Wall")


# Create the binding library
pybind11_add_module(cpp SHARED smoother_sycl/source.cpp)


target_link_libraries(cpp PUBLIC pybind11::module)
   
# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})