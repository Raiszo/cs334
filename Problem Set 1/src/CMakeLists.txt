file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(src ${HW1_files} ${hdr} ${cu})
# target_link_libraries(HW1 ${OpenCV_LIBS} )
