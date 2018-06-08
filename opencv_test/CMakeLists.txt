cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

# Important step for Windows users
# if (WIN32)
# 	SET("OpenCV_DIR" "C:/opencv_3.0/opencv/build")
# endif(WIN32)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
