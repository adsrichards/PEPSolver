cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TorchPEPS)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(TorchPEPS ${SOURCES})
target_link_libraries(TorchPEPS "${TORCH_LIBRARIES}")
set_property(TARGET TorchPEPS PROPERTY CXX_STANDARD 17)
