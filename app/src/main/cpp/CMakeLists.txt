cmake_minimum_required(VERSION 3.22.1)
project("threep_native" LANGUAGES CXX)

add_library(threep_native SHARED native-lib.cpp)
target_compile_features(threep_native PRIVATE cxx_std_17)
target_compile_options(threep_native PRIVATE -O3 -DNDEBUG -pthread)
target_link_options(threep_native PRIVATE "-Wl,-z,max-page-size=16384")

find_library(log-lib log)
target_link_libraries(threep_native PRIVATE ${log-lib})
