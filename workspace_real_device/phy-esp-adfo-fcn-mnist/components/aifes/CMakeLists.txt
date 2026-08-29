cmake_minimum_required(VERSION 3.22)

# This component is the AIfES library itself.
set(AIFES_DIR ${CMAKE_CURRENT_LIST_DIR})

# Collect sources. (Only AIfES code; CMSIS here looks header-only in your tree.)
file(GLOB_RECURSE AIFES_SRCS
    "${AIFES_DIR}/basic/*.c"
    "${AIFES_DIR}/cnn/*.c"
    "${AIFES_DIR}/core/*.c"
    "${AIFES_DIR}/custom_fcn/*.c" # inlcude customized function
    "${AIFES_DIR}/*.c"
    "${AIFES_DIR}/*.cpp"      # e.g., aifes_config.cpp
)

idf_component_register(
    SRCS         ${AIFES_SRCS}
    INCLUDE_DIRS
                 "${AIFES_DIR}"
                 "${AIFES_DIR}/basic"
                 "${AIFES_DIR}/cnn"
                 "${AIFES_DIR}/core"   # harmless if header-only; remove if not needed
                 "${AIFES_DIR}/custom_fcn"
    REQUIRES     # add other IDF components if you use them from AIfES (usually none)
)

# Public compile definitions (optional but handy for #ifdefs in your code)
target_compile_definitions(${COMPONENT_LIB} PUBLIC AIFES_PLATFORM_ESP_IDF=1)

# If you have any C++ sources (like aifes_config.cpp), ensure a sensible standard.
# IDF sets C++14 by default; bump here if you need newer.
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)

# You can also tune optimization flags for this component only (optional):
# target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-error=stringop-overflow
)
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-error=maybe-uninitialized
)

