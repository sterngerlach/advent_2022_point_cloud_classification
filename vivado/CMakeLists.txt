
# CMakeLists.txt

include(CMakeParseArguments)

cmake_minimum_required(VERSION 3.16)

project(pointnet_cls_vivado CXX)

# Check that Vivado is available (settings64.sh is sourced)
find_program(VIVADO_SOURCED vivado)

if (NOT VIVADO_SOURCED)
  message(FATAL_ERROR "vivado command is not available")
endif()

# Working directory for Vivado
set(VIVADO_WORK_DIR ${PROJECT_SOURCE_DIR}/work)

if (NOT EXISTS ${VIVADO_WORK_DIR})
  message(FATAL_ERROR "Working directory for Vivado does not exist: "
          ${VIVADO_WORK_DIR})
else()
  message(STATUS "Working directory for Vivado: ${VIVADO_WORK_DIR}")
endif()

# Working directory for Vitis HLS
set(HLS_WORK_DIR ${PROJECT_SOURCE_DIR}/../hls/work)

if (NOT EXISTS ${HLS_WORK_DIR})
  message(FATAL_ERROR "Working directory for Vitis HLS does not exist: "
          ${HLS_WORK_DIR})
else()
  message(STATUS "Working directory for Vitis HLS: ${HLS_WORK_DIR}")
endif()

# Output bitstream directory
set(BITSTREAM_DIR ${PROJECT_SOURCE_DIR}/bitstream)

if (NOT EXISTS ${BITSTREAM_DIR})
  message(FATAL_ERROR "Directory for output bitstream does not exist: "
          ${BITSTREAM_DIR})
else()
  message(STATUS "Directory for output bitstream: ${BITSTREAM_DIR}")
endif()

# Root directory for Vivado 2022.1
set(VIVADO_ROOT_DIR "/tools/Xilinx/Vivado/2022.1"
    CACHE STRING "Root directory for Vivado 2022.1")

if (NOT EXISTS ${VIVADO_ROOT_DIR})
  message(FATAL_ERROR "Root directory for Vivado 2022.1 does not exist: "
          ${VIVADO_ROOT_DIR})
else()
  message(STATUS "Root directory for Vivado 2022.1: ${VIVADO_ROOT_DIR}")
endif()

# Target board name (Xilinx ZCU104, Avnet Ultra96v2)
set(TARGET_BOARD "zcu104" CACHE STRING
    "Target device (Xilinx ZCU104: zcu104, Avnet Ultra96v2: ultra96v2)")

# Target device name
if (${TARGET_BOARD} STREQUAL "zcu104")
  set(TARGET_DEVICE "xczu7ev-ffvc1156-2-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
elseif (${TARGET_BOARD} STREQUAL "ultra96v2")
  set(TARGET_DEVICE "xczu3eg-sbva484-1-e")
  message(STATUS "Target board: ${TARGET_BOARD}")
  message(STATUS "Target device: ${TARGET_DEVICE}")
else()
  message(FATAL_ERROR "Unexpected target board: ${TARGET_BOARD}")
endif()

# Path to the Tcl scripts
set(TCL_CREATE_PROJECT_PATH ${PROJECT_SOURCE_DIR}/tcl/create_project.tcl)
set(TCL_IMPL_EXPORT_PATH ${PROJECT_SOURCE_DIR}/tcl/impl_export.tcl)

if (NOT EXISTS ${TCL_CREATE_PROJECT_PATH})
  message(FATAL_ERROR "Tcl script for creating project does not exist: "
          ${TCL_CREATE_PROJECT_PATH})
else()
  message(STATUS "Tcl script for creating project: "
          ${TCL_CREATE_PROJECT_PATH})
endif()

if (NOT EXISTS ${TCL_IMPL_EXPORT_PATH})
  message(FATAL_ERROR "Tcl script for implementation does not exist: "
          ${TCL_IMPL_EXPORT_PATH})
else()
  message(STATUS "Tcl script for implementation: "
          ${TCL_IMPL_EXPORT_PATH})
endif()

# Path to the Bash scripts
set(BASH_COPY_BITSTREAM_PATH ${PROJECT_SOURCE_DIR}/tcl/copy_bitstream.sh)

if (NOT EXISTS ${BASH_COPY_BITSTREAM_PATH})
  message(FATAL_ERROR "Bash script for copying bitstream does not exist: "
          ${BASH_COPY_BITSTREAM_PATH})
else()
  message(STATUS "Bash script for copying bitstream: "
          ${BASH_COPY_BITSTREAM_PATH})
endif()

function(vivado_add_targets project_name top_function_name
         strategy board_design_tcl)

  # Root directory for Vivado project
  set(vivado_project_dir ${VIVADO_WORK_DIR}/${project_name})
  message(STATUS "Root directory for Vivado project: ${vivado_project_dir}")

  # PointNet IP repository
  # Note: Project name must be the same for Vivado and Vitis HLS
  # Note: Vivado HLS must contain a solution called `solution`
  set(ip_repo_dir ${HLS_WORK_DIR}/${project_name}/solution/impl/ip)

  if (NOT EXISTS ${ip_repo_dir})
    message(FATAL_ERROR "PointNet IP repository does not exist: "
            ${ip_repo_dir})
  else()
    message(STATUS "PointNet IP repository: ${ip_repo_dir}")
  endif()

  # Check that the strategy is `runtime_optimized` or `default`
  if (${strategy} STREQUAL "runtime_optimized")
    message(STATUS "Strategy: runtime optimized")
  elseif (${strategy} STREQUAL "default")
    message(STATUS "Strategy: default")
  else()
    message(FATAL_ERROR "Unexpected strategy: ${strategy}")
  endif()

  # Check that the Tcl script for board design exists
  if (NOT EXISTS ${board_design_tcl})
    message(FATAL_ERROR "Tcl script for board design: ${board_design_tcl}")
  else()
    message(STATUS "Tcl script for board design: ${board_design_tcl}")
  endif()

  add_custom_target(${project_name}_create
    COMMAND vivado -mode batch -source ${TCL_CREATE_PROJECT_PATH}
    -tclargs ${vivado_project_dir} ${project_name} ${top_function_name}
    ${TARGET_DEVICE} ${ip_repo_dir} ${strategy} ${board_design_tcl})
  add_custom_target(${project_name}_impl
    COMMAND vivado -mode batch -source ${TCL_IMPL_EXPORT_PATH}
    -tclargs ${vivado_project_dir} ${project_name}.xpr)
  add_custom_target(${project_name}_copy_bitstream
    COMMAND bash ${BASH_COPY_BITSTREAM_PATH}
    ${vivado_project_dir} ${project_name}
    ${BITSTREAM_DIR} ${project_name})
endfunction()

# Project name and top function must be the same as the Vitis HLS one
# Refer to hls/CMakeLists.txt
vivado_add_targets(pointnet_naive PointNetClsTop
  runtime_optimized ${PROJECT_SOURCE_DIR}/tcl/board_design_naive.tcl)

vivado_add_targets(pointnet_naive_150 PointNetClsTop
  runtime_optimized ${PROJECT_SOURCE_DIR}/tcl/board_design_naive_150.tcl)

vivado_add_targets(pointnet_opt1 PointNetClsTop
  runtime_optimized ${PROJECT_SOURCE_DIR}/tcl/board_design_naive_150.tcl)

vivado_add_targets(pointnet_opt2 PointNetClsTop
  runtime_optimized ${PROJECT_SOURCE_DIR}/tcl/board_design_naive_150.tcl)

vivado_add_targets(pointnet_opt3 PointNetClsTop
  runtime_optimized ${PROJECT_SOURCE_DIR}/tcl/board_design_opt3_150.tcl)
