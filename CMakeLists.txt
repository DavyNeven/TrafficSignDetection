cmake_minimum_required(VERSION 3.0.2)
project(testCaffe)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(caffeinated_application main.cpp Classifier.cpp Classifier.h Detector.cpp Detector.h MultiScaleDetector.cpp MultiScaleDetector.h)
target_link_libraries(caffeinated_application ${Caffe_LIBRARIES})
set_target_properties(caffeinated_application PROPERTIES COMPILE_FLAGS "-D_LINUX -lX11 -D_REENTRANT -Wall  -O3 -march=native -msse2avx")