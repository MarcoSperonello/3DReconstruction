cmake_minimum_required(VERSION 2.8)

project( 3DReconstruction )

find_package( PCL 1.2 REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
add_executable( 3DReconstruction main.cpp )
target_link_libraries( 3DReconstruction ${OpenCV_LIBS} ${PCL_LIBRARIES})
