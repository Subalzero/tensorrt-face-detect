cmake_minimum_required(VERSION 3.22)

project(FaceDetection)

include_directories( ${OpenCV_INCLUDE_DIRS} ${TENSORRT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include )
link_directories( ${OpenCV_LIB_DIR} ${TENSORRT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 )

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB COMMON_H_FILES "tensorrt_samples/common/*.h")
file(GLOB COMMON_SRC_FILES "tensorrt_samples/common/*.cpp" "tensorrt_samples/common/*.c")
file(GLOB UTILS_H_FILES "tensorrt_samples/utils/*.h")
file(GLOB UTILS_SRC_FILES "tensorrt_samples/utils/*.cpp")

file(GLOB INCLUDE_FILES include/*.hpp include/*.h)
file(GLOB SRC_FILES src/*.cpp src/*.c)

add_executable(FaceDetection main.cpp ${INCLUDE_FILES} ${SRC_FILES} ${COMMON_H_FILES} ${COMMON_SRC_FILES} ${UTILS_H_FILES} ${UTILS_SRC_FILES})

target_include_directories(FaceDetection 
PRIVATE 
    include
    tensorrt_samples/common
    tensorrt_samples
)

target_link_libraries(FaceDetection 
PRIVATE
    nvinfer_10
    nvonnxparser_10
    nvinfer_plugin_10
    cudart
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

add_custom_command(
    TARGET FaceDetection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    ${CMAKE_CURRENT_BINARY_DIR}/models
)

if (WIN32)
    
    add_custom_command(
        TARGET FaceDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TENSORRT_DIR}/lib
        ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET FaceDetection PROPERTY CXX_STANDARD 20)
endif()