# The following lines of boilerplate have to be in your project's CMakeLists
# in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

message(STATUS "DATASET ${DATASET}")

if(DEFINED DATASET)
    # Set it as a cache variable so it's visible to all subdirectories
    set(ENV{ESP_DATASET} ${DATASET})
endif()

set(EXTRA_COMPONENT_DIRS $ENV{IDF_PATH}/components)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=maybe-uninitialized" APPEND)
project(esp-adfo)
