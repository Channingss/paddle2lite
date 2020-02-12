if (LITE_ON_MODEL_OPTIMIZE_TOOL)
    message(STATUS "Compiling py_optimize_tool")
    message(STATUS "pybind11_python :" ${PYBIND11_PYTHON_VERSION})
    MESSAGE(STATUS "python version:" )
    MESSAGE(STATUS ${PYTHON_EXECUTABLE} )
    MESSAGE(STATUS ${PYTHON_VERSION_STRING} )
    MESSAGE(STATUS ${CMAKE_SYSTEM_PROCESSOR})
    lite_cc_library(lite SHARED 
        SRCS pybind.cc ../../cxx_api_impl.cc ../../cxx_api_impl.cc ../../paddle_api.cc ../../cxx_api.cc
        DEPS kernel op optimizer mir_passes utils python pybind)
    add_dependencies(lite op_list_h kernel_list_h all_kernel_faked_cc)
endif(LITE_ON_MODEL_OPTIMIZE_TOOL)


