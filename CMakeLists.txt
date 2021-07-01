cmake_minimum_required(VERSION 3.10)
project(test1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/media/dp/DATA/huihua_robot/libtorch-cxx11-abi-shared-with-deps-1.8.1+cu102/libtorch")

# OpenCV
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(test1 main.cpp Configs.h TorchScript.cpp TorchScript.h)

target_link_libraries(test1 ${OpenCV_LIBS} ${TORCH_LIBRARIES})