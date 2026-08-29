# The following lines of boilerplate have to be in your project's CMakeLists
# in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

message(STATUS "DATASET ${DATASET}")
message(STATUS "METHOD" ${METHOD})

if(DEFINED DATASET)
    # Set it as a cache variable so it's visible to all subdirectories
    set(ENV{ESP_DATASET} ${DATASET})
endif()

if(DEFINED METHOD)
    # Set it as a cache variable so it's visible to all subdirectories
    set(ENV{ESP_METHOD} ${METHOD})
endif()

if(DEFINED MAXIMUM_NUM_DEVICES)
    # Set it as a cache variable so it's visible to all subdirectories
    set(ENV{ESP_MAXIMUM_NUM_DEVICES} ${MAXIMUM_NUM_DEVICES})
endif()

set(EXTRA_COMPONENT_DIRS $ENV{IDF_PATH}/components)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=maybe-uninitialized" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-error=stringop-overflow" APPEND)
project(esp-adfo)
