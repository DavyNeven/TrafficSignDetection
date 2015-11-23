cmake_minimum_required(VERSION 3.0.2)
project(testCaffe)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(caffeinated_application main.cpp Classifier.cpp Classifier.h Detector.cpp Detector.h)
target_link_libraries(caffeinated_application ${Caffe_LIBRARIES})