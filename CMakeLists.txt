cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PEPSolver)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")

file(GLOB SOURCES "src/*.cpp")

add_executable(PEPSolver ${SOURCES})
target_link_libraries(PEPSolver "${TORCH_LIBRARIES}")
set_property(TARGET PEPSolver PROPERTY CXX_STANDARD 17)
