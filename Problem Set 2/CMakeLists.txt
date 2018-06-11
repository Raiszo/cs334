cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(HW1)

# Important step for Windows users
# if (WIN32)
# 	SET("OpenCV_DIR" "C:/opencv_3.0/opencv/build/")
# endif(WIN32)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")



if (CUDA_FOUND AND OpenCV_FOUND)

	add_subdirectory(src)
	
else(CUDA_FOUND AND OpenCV_FOUND)
  message("CUDA or OpenCV are not installed on this system.")
endif()
