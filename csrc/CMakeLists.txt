# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep LANGUAGES MUSA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(MUSA_SEPARABLE_COMPILATION ON)
list(APPEND MUSA_MCC_FLAGS "-O3")
list(APPEND MUSA_MCC_FLAGS "--register-usage-level=10,--warn-on-local-memory-usage")

set(TORCH_MUSA_ARCH_LIST "3.1")
find_package(MUSAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(MTSHMEM REQUIRED HINTS ${MTSHMEM_ROOT_DIR}/lib/cmake/mtshmem)

add_library(mtshmem ALIAS mtshmem::mtshmem)
add_library(mtshmem_host ALIAS mtshmem::mtshmem_host)
add_library(mtshmem_device ALIAS mtshmem::mtshmem_device)

# Seems bugs with CMake, MCC 12 and C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MUSA_STANDARD 14)

include_directories(${MUSA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${MTSHMEM_INCLUDE_DIR})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${MUSA_TOOLKIT_ROOT_DIR}/lib ${MTSHMEM_LIB_DIR})

add_subdirectory(kernels)

# Link CPP and MUSA together
pybind11_add_module(deep_ep_cpp deep_ep.cpp)
target_link_libraries(deep_ep_cpp PRIVATE ${EP_MUSA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
