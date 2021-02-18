cmake_minimum_required(VERSION 2.8.3)
project(lanms)

set(CMAKE_CXX_STANDARD 11)
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else(WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-pthread)
endif(WIN32)

find_package (Python3 REQUIRED COMPONENTS Interpreter Development)

set(SOURCES "lanms.cpp" "adaptor.cpp" "include/clipper/clipper.cpp")
add_library(lanms_library SHARED ${SOURCES})
target_link_libraries(lanms_library ${Python3_LIBRARIES})
target_include_directories(lanms_library PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${Python3_INCLUDE_DIRS})

include_directories("include")
include_directories(${Python3_INCLUDE_DIRS})

add_executable(main main.cpp ${SOURCES})
target_link_libraries(main lanms_library)
