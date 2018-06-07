
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(HW1)

# Important step for Windows users
if (WIN32)
	SET("OpenCV_DIR" "C:/opencv_3.0/opencv/build/")
endif(WIN32)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(HW1 ${OpenCV_LIBS} )


if(CUDA_FOUND)
	set(CUDA_NVCC_FLAGS "
  -gencode;arch=compute_61,code=sm_61;")

	# add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

	add_subdirectory(src)
	
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
