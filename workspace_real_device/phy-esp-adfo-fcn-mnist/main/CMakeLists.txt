# Set the source files for the main component

set(SOURCES "esp-adfo.c"
            "adfo-model.c"
            "adfo-com.c"
)

# Add the include directories
set(INCLUDE_DIRS "." "dataset_mnist/")


if(DEFINED ENV{ESP_DATASET})
    set(DATASET $ENV{ESP_DATASET})
    message(STATUS "Building with dataset: ${DATASET}")

    # Add dataset source file dynamically
    list(APPEND INCLUDE_DIRS "dataset_mnist/device_${DATASET}")

else()
    message(FATAL_ERROR "No dataset file specified. Default dataset will be used.")
endif()

# Add the include directories for AIfES
list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

# Register the component with the specified sources and include directories
idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS ${INCLUDE_DIRS}
    REQUIRES       aifes    # link against the AIfES component
    PRIV_REQUIRES esp_timer nvs_flash esp_netif esp_http_server esp_wifi esp_eth
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=maybe-uninitialized)
target_compile_definitions(${COMPONENT_LIB} PRIVATE "DATASET=${DATASET}")
