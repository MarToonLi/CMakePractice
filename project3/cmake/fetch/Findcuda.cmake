list(APPEND CMAKE_MESSAGE_INDENT "[cuda-nvml]  ")

set(cuda_INCLUDE_DIR        ${cuda_ROOT}/include)
set(cuda_LIBRARY            ${cuda_ROOT}/lib/x64/nvml.lib)
set(ENV{cuda_INCLUDE_DIR}        ${cuda_ROOT}/include)
set(ENV{cuda_LIBRARY}            ${cuda_ROOT}/lib/x64/nvml.lib)
set(cuda_FOUND 1)


message("cuda_FOUND:       ${cuda_FOUND}")
message("cuda_ROOT:        ${cuda_ROOT}")
message("cuda_INCLUDE_DIR: ${cuda_INCLUDE_DIR}")
message("cuda_LIBRARY:     ${cuda_LIBRARY}")



include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(cuda 
  REQUIRED_VARS cuda_LIBRARY cuda_INCLUDE_DIR 
  VERSION_VAR cuda_VERSION)

if(onnxruntime_FOUND)
  set(cuda_INCLUDE_DIRS ${cuda_INCLUDE_DIR})
  set(cuda_LIBRARIES    ${cuda_LIBRARY})

  add_library(cuda SHARED IMPORTED)
  target_include_directories(cuda INTERFACE ${cuda_INCLUDE_DIRS})
  if(WIN32)
    set_target_properties(cuda PROPERTIES 
      IMPORTED_IMPLIB "${cuda_LIBRARY}")
  else()
    set_target_properties(cuda PROPERTIES 
      IMPORTED_LOCATION "${cuda_LIBRARY}")
  endif()
endif()


list(POP_BACK CMAKE_MESSAGE_INDENT)
