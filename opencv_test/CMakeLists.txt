cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

# Important step for Windows users
# if (WIN32)
# 	SET("OpenCV_DIR" "C:/opencv_3.0/opencv/build/x64/vc15/bin")
# endif(WIN32)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
