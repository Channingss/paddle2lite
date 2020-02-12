if (NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
    return()
endif()
set(CMAKE_MODULE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS ${CMAKE_MODULE_PATH})
include(python)    # download, build, install python
include(pybind11)    # download, build, install pybind11
add_subdirectory(pybind)
