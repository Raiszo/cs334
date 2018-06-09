find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )


set(CUDA_NVCC_FLAGS "
  -gencode;arch=compute_61,code=sm_61;")

# add -Wextra compiler flag for gcc compilations
if (UNIX)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
endif (UNIX)

# add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")



file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(rgb2grey ${HW1_files} ${hdr} ${cu})
target_link_libraries(rgb2grey ${OpenCV_LIBS} )
