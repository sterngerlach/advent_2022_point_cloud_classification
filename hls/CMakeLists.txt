
# CMakeLists.txt

include(CMakeParseArguments)

cmake_minimum_required(VERSION 3.16)

project(pointnet_cls CXX)

# Find HDF5 library
find_package(HDF5 REQUIRED COMPONENTS CXX HL)

# Check that Vitis HLS is available (settings64.sh is sourced)
find_program(VITIS_HLS_SOURCED vitis_hls)

if (NOT VITIS_HLS_SOURCED)
  message(FATAL_ERROR "vitis_hls command is not available")
endif()

# Include directories
set(POINTNET_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)

# Working directory
set(POINTNET_WORK_DIR ${PROJECT_SOURCE_DIR}/work)

if (NOT EXISTS ${POINTNET_WORK_DIR})
  message(FATAL_ERROR "Working directory does not exist: "
          ${POINTNET_WORK_DIR})
else()
  message(STATUS "Working directory: ${POINTNET_WORK_DIR}")
endif()

# Root directory for Vitis HLS 2022.1
set(VITIS_HLS_ROOT_DIR "/tools/Xilinx/Vitis_HLS/2022.1"
    CACHE STRING "Root directory for Vitis HLS 2022.1")
# Include directories for Vitis HLS 2022.1
set(VITIS_HLS_INCLUDE_DIRS ${VITIS_HLS_ROOT_DIR}/include)

if (NOT EXISTS ${VITIS_HLS_ROOT_DIR})
  message(FATAL_ERROR "Root directory for Vitis HLS 2022.1 does not exist: "
          ${VITIS_HLS_ROOT_DIR})
else()
  message(STATUS "Root directory for Vitis HLS 2022.1: "
          ${VITIS_HLS_ROOT_DIR})
endif()

if (NOT EXISTS ${VITIS_HLS_INCLUDE_DIRS})
  message(FATAL_ERROR "Include directory for Vitis HLS 2022.1 does not exist: "
          ${VITIS_HLS_INCLUDE_DIRS})
else()
  message(STATUS "Include directory for Vitis HLS 2022.1: "
          ${VITIS_HLS_INCLUDE_DIRS})
endif()

# Target board name (Xilinx ZCU104, Avnet Ultra96v2)
set(TARGET_BOARD "zcu104" CACHE STRING
    "Target device (zcu104, ultra96v2, pynqz2)")

# Target device name
if (${TARGET_BOARD} STREQUAL "zcu104")
  set(TARGET_DEVICE "xczu7ev-ffvc1156-2-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
elseif (${TARGET_BOARD} STREQUAL "ultra96v2")
  set(TARGET_DEVICE "xczu3eg-sbva484-1-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
elseif (${TARGET_BOARD} STREQUAL "pynqz2")
  set(TARGET_DEVICE "xc7z020-clg400-1")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
else()
  message(FATAL_ERROR "Unexpected target board: ${TARGET_BOARD}")
endif()

# Path to the Tcl scripts
set(VITIS_HLS_TCL_PATH ${PROJECT_SOURCE_DIR}/tcl/common.tcl)

if (NOT EXISTS ${VITIS_HLS_TCL_PATH})
  message(FATAL_ERROR "Tcl script does not exist: ${VITIS_HLS_TCL_PATH}")
else()
  message(STATUS "Tcl script found: ${VITIS_HLS_TCL_PATH}")
endif()

macro(prefix_options srcs option dsts)
  foreach(src ${srcs})
    list(APPEND ${dsts} "${option}${src}")
  endforeach()
endmacro()

macro(collect_lib_directories libs dsts)
  foreach(lib ${libs})
    get_filename_component(lib_dir ${lib} DIRECTORY)
    if (NOT lib_dir STREQUAL "")
      list(APPEND ${dsts} ${lib_dir})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES ${dsts})
endmacro()

macro(collect_lib_names libs dsts)
  foreach(lib ${libs})
    get_filename_component(lib_name ${lib} NAME_WE)
    string(REGEX REPLACE "^lib*" "" lib_name "${lib_name}")
    list(APPEND ${dsts} ${lib_name})
  endforeach()
  list(REMOVE_DUPLICATES ${dsts})
endmacro()

# HDF5 library directories (e.g., /usr/lib/x86_64-linux-gnu/hdf5-serial)
collect_lib_directories("${HDF5_LIBRARIES}" hdf5_lib_dirs)
collect_lib_directories("${HDF5_HL_LIBRARIES}" hdf5_lib_dirs)
message(STATUS "HDF5 library directories: ${hdf5_lib_dirs}")

# HDF5 library names (e.g., hdf5, hdf5_cpp, hdf5_hl_cpp)
collect_lib_names("${HDF5_LIBRARIES}" hdf5_lib_names)
collect_lib_names("${HDF5_HL_LIBRARIES}" hdf5_lib_names)
message(STATUS "HDF5 library names: ${hdf5_lib_names}")

function(hls_add_targets project_name top_function_name)
  cmake_parse_arguments(ARG "" "CLOCK_PERIOD"
    "HLS_SRCS;TB_SRCS;CXXFLAGS" ${ARGN})

  set(include_dirs ${POINTNET_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
  prefix_options("${include_dirs}" "-I" include_options)
  message(STATUS "Include directories: ${include_options}")

  set(library_dirs ${hdf5_lib_dirs})
  list(REMOVE_DUPLICATES library_dirs)
  prefix_options("${library_dirs}" "-L" library_dir_options)
  prefix_options("${library_dirs}" "-Wl,-rpath," rpath_options)
  message(STATUS "Library directories: ${library_dir_options}")

  set(libraries "m" "pthread" "dl" "stdc++fs" ${hdf5_lib_names})
  list(REMOVE_DUPLICATES libraries)
  prefix_options("${libraries}" "-l" library_options)
  message(STATUS "Library options: ${library_options}")

  set(hls_cxx_flags ${ARG_CXXFLAGS} ${include_options})
  string(REPLACE ";" " " hls_cxx_flags_str "${hls_cxx_flags}")

  set(hls_linker_flags ${library_dir_options}
      ${library_options} ${rpath_options})
  string(REPLACE ";" " " hls_linker_flags_str "${hls_linker_flags}")

  set(hls_testbench_cxx_flags "-O3")
  string(REPLACE ";" " " hls_testbench_cxx_flags_str
         "${hls_testbench_cxx_flags}")

  string(REPLACE ";" " " hls_srcs_str "${ARG_HLS_SRCS}")
  string(REPLACE ";" " " tb_srcs_str "${ARG_TB_SRCS}")
  message(STATUS "HLS source files: ${hls_srcs_str}")
  message(STATUS "HLS testbench files: ${tb_srcs_str}")

  list(LENGTH ARG_HLS_SRCS num_hls_srcs)
  list(LENGTH ARG_TB_SRCS num_tb_srcs)

  if (${num_tb_srcs} GREATER 0)
    message(STATUS "Appending a custom target: ${project_name}_csim")
    add_custom_target(${project_name}_csim
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csim" ${project_name} ${top_function_name}
        ${TARGET_DEVICE} ${ARG_CLOCK_PERIOD}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
        "linker_flags=${hls_linker_flags_str}" "${tb_srcs_str}"
        "testbench_cxx_flags=${hls_testbench_cxx_flags_str}"
      WORKING_DIRECTORY ${POINTNET_WORK_DIR})
  endif()

  if (${num_hls_srcs} GREATER 0)
    message(STATUS "Appending a custom target: ${project_name}_csynth")
    add_custom_target(${project_name}_csynth
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csynth" ${project_name} ${top_function_name}
        ${TARGET_DEVICE} ${ARG_CLOCK_PERIOD}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
      WORKING_DIRECTORY ${POINTNET_WORK_DIR})

    message(STATUS "Appending a custom target: ${project_name}_csynth_export")
    add_custom_target(${project_name}_csynth_export
      COMMAND vitis_hls -f ${VITIS_HLS_TCL_PATH}
        "csynth_export" ${project_name} ${top_function_name}
        ${TARGET_DEVICE} ${ARG_CLOCK_PERIOD}
        "cxx_flags=${hls_cxx_flags_str}" "${hls_srcs_str}"
      WORKING_DIRECTORY ${POINTNET_WORK_DIR})
  endif()
endfunction()

hls_add_targets(pointnet_naive_fail PointNetClsTop
  CLOCK_PERIOD 10.0
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_naive_fail.cpp
  CXXFLAGS "-DVALUE_BIT_WIDTH=32 -DVALUE_INT_WIDTH=16 "
    "-DPARAM_BIT_WIDTH=32 -DPARAM_INT_WIDTH=16")

hls_add_targets(pointnet_naive PointNetClsTop
  CLOCK_PERIOD 10.0
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_naive.cpp
  CXXFLAGS "-DVALUE_BIT_WIDTH=32 -DVALUE_INT_WIDTH=16 "
    "-DPARAM_BIT_WIDTH=32 -DPARAM_INT_WIDTH=16")

hls_add_targets(pointnet_naive_250 PointNetClsTop
  CLOCK_PERIOD 4.0
  HLS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/top_naive.cpp
  CXXFLAGS "-DVALUE_BIT_WIDTH=32 -DVALUE_INT_WIDTH=16 "
    "-DPARAM_BIT_WIDTH=32 -DPARAM_INT_WIDTH=16")
