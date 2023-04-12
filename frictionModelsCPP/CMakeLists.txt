cmake_minimum_required(VERSION 3.0.0)
project(testcpp VERSION 0.1.0)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

include(CTest)
enable_testing()

find_package(PythonLibs REQUIRED)
include_directories(/home/gabriel/anaconda3/envs/frictionModeling/include/python3.10)
find_package(pybind11 CONFIG)
set(PYBIND11_PYTHON_VERSION "3.10") 

# Add the source files to a library
add_library(mymodule_lib SHARED utils.cpp)

pybind11_add_module(FrictionModelCPPClass SHARED fullModel.cpp)

target_link_libraries(FrictionModelCPPClass PRIVATE mymodule_lib pybind11::module)

add_library(mymodule_lib2 SHARED fullModel.cpp)

pybind11_add_module(ReducedFrictionModelCPPClass SHARED reducedModel.cpp)

target_link_libraries(ReducedFrictionModelCPPClass PRIVATE mymodule_lib mymodule_lib2 pybind11::module)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
