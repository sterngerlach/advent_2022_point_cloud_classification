
# common.tcl

# vitis_hls -f common.tcl <Command> <Project Name> <Target Device>
# <Compiler Flags> <Sources>
# <Linker Flags> <Testbench> <Testbench Compiler Additional Flags>
# <Testbench Command-line Arguments>

if {$argc < 9} {
  puts { Options: <Command> <Project Name> <Top>\
         <Target Device> <Clock Period>\
         <Compiler Flags> <Sources>\
         <Linker Flags> <Testbench> <Testbench Compiler Additional Flags>\
         <Testbench Command-line Arguments> }
  exit 2
}

# `[lindex $argv 0]` is `-f`
# `[lindex $argv 1]` is `common.tcl`

# Get command-line options
set command [lindex $argv 2]
set project_name [lindex $argv 3]
set top_function_name [lindex $argv 4]
set target_device_name [lindex $argv 5]
set clock_period [lindex $argv 6]
set hls_cxx_flags [lindex $argv 7]
set source_files [lindex $argv 8]
set hls_linker_flags [lindex $argv 9]
set testbench_files [lindex $argv 10]
set testbench_cxx_flags [lindex $argv 11]
set testbench_argv [lindex $argv 12]

regsub "cxx_flags=" $hls_cxx_flags "" hls_cxx_flags
regsub "linker_flags=" $hls_linker_flags "" hls_linker_flags
regsub "testbench_cxx_flags=" $testbench_cxx_flags "" testbench_cxx_flags

set source_cxx_flags "${hls_cxx_flags} -Wno-unknown-pragmas -std=c++14"
set testbench_cxx_flags "${hls_cxx_flags} ${testbench_cxx_flags} \
  -Wno-unknown-pragmas -std=c++14"

puts "HLS compiler flags: ${hls_cxx_flags}"
puts "HLS linker flags: ${hls_linker_flags}"
puts "HLS compiler flags for source: ${source_cxx_flags}"
puts "HLS compiler flags for testbench: ${testbench_cxx_flags}"
puts "HLS source files: ${source_files}"
puts "HLS testbench files: ${testbench_files}"

open_project -reset ${project_name}

set_top ${top_function_name}
add_files ${source_files} -cflags ${source_cxx_flags}

open_solution "solution" -flow_target vivado
set_part ${target_device_name}
create_clock -period ${clock_period} -name default

config_export -description {An IP for PointNet classification} \
  -format ip_catalog -rtl verilog -vendor sterngerlach

# We need to set `m_axi_alignment_byte_size` to 64 to automatically widen
# the port width
# config_interface -m_axi_alignment_byte_size 64

if {${command} == "csim"} {
  # C-simulation
  add_files -tb ${testbench_files} \
    -cflags ${testbench_cxx_flags} -csimflags ${testbench_cxx_flags}
  csim_design -clean -O \
    -ldflags ${hls_linker_flags} -argv ${testbench_argv}
} elseif {${command} == "csynth"} {
  # C synthesis
  csynth_design
} elseif {${command} == "csynth_export"} {
  # C synthesis and then export design for Vivado
  csynth_design
  # Specify VLNV (Vendor:Library:Name:Version)
  # Use -vendor, -library, -ipname, -version options
  export_design -description {An IP for PointNet classification} \
    -display_name ${top_function_name} \
    -format ip_catalog -rtl verilog \
    -vendor sterngerlach -library hls \
    -ipname ${top_function_name} -version 1.0
} else {
  puts "Unknown command: ${command}"
  exit 2
}

exit 0
