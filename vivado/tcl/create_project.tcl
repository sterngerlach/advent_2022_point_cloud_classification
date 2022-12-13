
# create_project.tcl

# References
# https://qiita.com/ikwzm/items/666dcf3b90c36d16a0ed
# https://qiita.com/ikwzm/items/36e5911c155c8303b705
# https://github.com/Digilent/ZYBO/blob/master/Projects/hdmi_in/proj/create_project.tcl
# https://github.com/Digilent/ZYBO/blob/master/Projects/hdmi_in/src/bd/system.tcl
# https://support.xilinx.com/s/article/62015?language=en_US
# https://fpga.kice.tokyo/fpga/synth-cmd

# vivado -mode batch -source create_project.tcl
# -tclargs <Project Directory> <Project Name> <Top> <Target Device>
# <IP Repository> <Strategy> <Board Design Tcl>

# <Strategy>: `runtime_optimized` or `default`

if {$argc < 7} {
  puts { Options: <Project Directory> <Project Name> <Top> <Target Device>
         <IP Repository> <Strategy> <Board Design Tcl> }
  puts { <Strategy>: `runtime_optimized` or `default` }
  exit 2
}

# Get command-line options
set project_dir [lindex $argv 0]
set project_name [lindex $argv 1]
set top_function_name [lindex $argv 2]
set device_part [lindex $argv 3]
set pointnet_ip_repo_dir [lindex $argv 4]
set strategy [lindex $argv 5]
set board_design_tcl [lindex $argv 6]

set project_dir [file normalize $project_dir]
set pointnet_ip_repo_dir [file normalize $pointnet_ip_repo_dir]

puts "Project directory: ${project_dir}"
puts "Project name: ${project_name}"
puts "Device part: ${device_part}"
puts "IP repository: ${pointnet_ip_repo_dir}"
puts "Strategy: ${strategy}"
puts "Board design Tcl: ${board_design_tcl}"

# Check that the project directory exists
set parent_dir [file dirname $project_dir]
if {![file isdirectory $parent_dir]} {
  puts "Parent directory does not exist: $parent_dir"
  exit 2
}

# Check that the Tcl script for board design exists
if {![file isfile $board_design_tcl]} {
  puts "Tcl script for board design does not exist: $board_design_tcl"
  exit 2
}

# Create a project
create_project -force $project_name $project_dir

# Set project properties
set proj [get_projects $project_name]
set_property "part" $device_part $proj
set_property "simulator_language" "Mixed" $proj
set_property "target_language" "VHDL" $proj
set_property "default_lib" "xil_defaultlib" $proj
# set_property "board_part" $board_part $proj

# Create `sources_1` fileset (if not found)
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

# Create `constrs_1` fileset (if not found)
if {[string equal [get_filesets -quiet constrs_1] ""]} {
  create_fileset -constrset constrs_1
}

# Create `sim_1` fileset (if not found)
# if {[string equal [get_filesets -quiet sim_1] ""]} {
#   create_fileset -simset sim_1
# }

# Set IP repository paths
set sources [get_filesets sources_1]
set_property "ip_repo_paths" $pointnet_ip_repo_dir $sources

# Refresh IP repositories
update_ip_catalog

# Create `synth_1` run (if not found)
set synth_1_flow "Vivado Synthesis 2022"
set synth_1_strategy "Vivado Synthesis Defaults"

if {[string equal [get_runs -quiet synth_1] ""]} {
  create_run -name synth_1 -part $device_part \
    -flow $synth_1_flow -strategy $synth_1_strategy -constrset constrs_1
} else {
  set_property flow $synth_1_flow [get_runs synth_1]
  set_property strategy $synth_1_strategy [get_runs synth_1]
}

# Set properties for `synth_1`
set_property "part" $device_part [get_runs synth_1]

if {[string equal $strategy "runtime_optimized"]} {
  # Optimize the runtime
  set_property "steps.synth_design.args.flatten_hierarchy" \
    none [get_runs synth_1]
  set_property "steps.synth_design.args.directive" \
    RuntimeOptimized [get_runs synth_1]
  set_property "steps.synth_design.args.fsm_extraction" \
    off [get_runs synth_1]
} elseif {[string equal $strategy "default"]} {
  # Default settings
  set_property "steps.synth_design.args.flatten_hierarchy" \
    rebuilt [get_runs synth_1]
  set_property "steps.synth_design.args.directive" \
    Default [get_runs synth_1]
  set_property "steps.synth_design.args.fsm_extraction" \
    auto [get_runs synth_1]
}

# Set the current synthesis run
current_run -synthesis [get_runs synth_1]

# Create `impl_1` run (if not found)
set impl_1_flow "Vivado Implementation 2022"
set impl_1_strategy "Vivado Implementation Defaults"

if {[string equal [get_runs -quiet impl_1] ""]} {
  create_run -name impl_1 -part $device_part \
    -flow $impl_1_flow -strategy $impl_1_strategy \
    -constrset constrs_1 -parent_run synth_1
} else {
  set_property flow $impl_1_flow [get_runs impl_1]
  set_property strategy $impl_1_strategy [get_runs impl_1]
}

# Set properties for `impl_1`
set_property "part" $device_part [get_runs impl_1]

if {[string equal $strategy "runtime_optimized"]} {
  # Optimize the runtime
  set_property "steps.opt_design.args.directive" \
    RuntimeOptimized [get_runs impl_1]
  set_property "steps.place_design.args.directive" \
    RuntimeOptimized [get_runs impl_1]
  set_property "steps.phys_opt_design.args.directive" \
    RuntimeOptimized [get_runs impl_1]
  set_property "steps.route_design.args.directive" \
    RuntimeOptimized [get_runs impl_1]
} elseif {[string equal $strategy "default"]} {
  # Default settings
  set_property "steps.opt_design.args.directive" \
    Default [get_runs impl_1]
  set_property "steps.place_design.args.directive" \
    Default [get_runs impl_1]
  set_property "steps.phys_opt_design.args.directive" \
    Default [get_runs impl_1]
  set_property "steps.route_design.args.directive" \
    Default [get_runs impl_1]
}

# Set the current implementation run
current_run -implementation [get_runs impl_1]

puts "Project successfully created: ${project_name}"

# Construct VLNV (Vendor:Library:Name:Version)
# Refer to hls/tcl/common.tcl
set pointnet_ip_vlnv "sterngerlach:hls:${top_function_name}:1.0"
puts "VLNV for PointNet IP core: ${pointnet_ip_vlnv}"
# Setup $argc and $argv before source
set argv [list $pointnet_ip_vlnv]
set argc 1
# Create the block design
source $board_design_tcl

# Design name must be `design0` (Refer to board_design.tcl)
# `get_bd_designs` returns multiple board designs (e.g., `bd_0d72` and
# `design0`) since we use AXI SmartConnect IP
set design_name [get_bd_designs -patterns "design0"]
if {[llength $design_name] < 1} {
  puts "Project does not contain the board design `design0`"
  exit 2
} elseif {[llength $design_name] > 1} {
  puts "Project contains multiple board designs starting with `design0`"
  exit 2
}

# Generate the HDL wrapper
add_files -norecurse -fileset sources_1 \
  [make_wrapper -files [get_files $design_name.bd] -top -force]

set_property "top" "${design_name}_wrapper" [get_filesets sources_1]

puts "Block design successfully created: $design_name.bd"

# Close the project
close_project

exit 0
