if(NOT TARGET slogging)
add_library(slogging INTERFACE)
target_sources(slogging INTERFACE slogger.h)
target_link_libraries(slogging INTERFACE spdlog)
target_include_directories(slogging INTERFACE ./)
endif()


find_package(OpenCV)
add_library(scommon INTERFACE)
target_sources(scommon INTERFACE scommon.h scommon.cpp)
target_include_directories(scommon INTERFACE ./)
target_link_libraries(scommon
INTERFACE
	opencv_world
)
