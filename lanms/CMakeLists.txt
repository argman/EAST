cmake_minimum_required(VERSION 3.4)
project(lanms)

set(CMAKE_CXX_STANDARD 11)
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else(WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-pthread)
endif(WIN32)

set(SOURCES "lanms.cpp" "include/clipper/clipper.cpp")

add_library(lanms_library SHARED ${SOURCES})
target_link_libraries(lanms_library)
target_include_directories(lanms_library PUBLIC ${CMAKE_CURRENT_LIST_DIR})

include_directories("include")

add_executable(main main.cpp ${SOURCES})
target_link_libraries(main)

add_subdirectory("pybind11")
pybind11_add_module(adaptor "adaptor.cpp" ${SOURCES})
target_link_libraries(adaptor PRIVATE lanms_library)
target_compile_definitions(adaptor PRIVATE)
