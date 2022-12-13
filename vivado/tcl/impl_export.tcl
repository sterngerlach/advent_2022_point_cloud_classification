
# impl_export.tcl

# vivado -mode batch -source impl_export.tcl
# -tclargs <Project Directory> <Project File Name>

# Example: vivado -mode batch -source ../tcl/impl_export.tcl \
# -tclargs ../work/test_tcl_script test_tcl_script.xpr

if {$argc < 2} {
  puts { Usage: <Project Directory> <Project File Name> }
  exit 2
}

# Get command-line options
set project_dir [lindex $argv 0]
set project_file_name [lindex $argv 1]

set project_dir [file normalize $project_dir]

puts "Project directory: ${project_dir}"
puts "Project file name: ${project_file_name}"

# Check that the project directory exists
if {![file isdirectory $project_dir]} {
  puts "Project directory does not exist: $project_dir"
  exit 2
}

set project_file_path [file join $project_dir $project_file_name]

# Check that the project file exists
if {![file isfile $project_file_path]} {
  puts "Project file does not exist: $project_file_path"
  exit 2
}

# Open the project
open_project $project_file_path

# Run synthesis
# Name should be `synth_1` (Refer to create_project.tcl)
launch_runs synth_1 -jobs 8 -verbose
wait_on_run synth_1

# Run implementation
# Name should be `impl_1` (Refer to create_project.tcl)
launch_runs impl_1 -jobs 8 -verbose
wait_on_run impl_1
open_run impl_1

report_utilization -file [file join $project_dir "utilization.rpt"]
report_power -file [file join $project_dir "power.rpt"]

# Write the bitstream file
launch_runs impl_1 -to_step write_bitstream -jobs 8 -verbose
wait_on_run impl_1

# Close the project
close_project

exit 0
