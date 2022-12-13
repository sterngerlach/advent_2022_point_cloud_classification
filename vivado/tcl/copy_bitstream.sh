#!/bin/bash
# copy_bitstream.sh

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <Project Directory> <Project Name>" \
       "<Output Directory> <Output File Name>"
  exit 1
fi

# Get command-line arguments
project_dir=$1
project_name=$2
out_dir=$3
out_name=$4

# Check that the project directory exists
if [[ ! -d $project_dir ]]; then
  echo "Project directory does not exist: $project_dir"
  exit 1
fi

# Note: implementation run should be `impl_1` (Refer to create_project.tcl)
# Note: source fileset should be `sources_1` (Refer to create_project.tcl)
# Note: design name should be `design0` (Refer to board_design.tcl)

# Path to the bitstream (*.bit)
impl_dir=$project_dir/$project_name.runs/impl_1
bitstream_path=$impl_dir/design0_wrapper.bit
# Path to the hardware handoff (*.hwh)
handoff_dir=$project_dir/$project_name.gen/sources_1/bd/design0/hw_handoff
handoff_path=$handoff_dir/design0.hwh

if [[ ! -f $bitstream_path ]]; then
  echo "Bitstream does not exist: $bitstream_path"
  exit 1
fi

if [[ ! -f $handoff_path ]]; then
  echo "Hardware handoff does not exist: $handoff_path"
  exit 1
fi

# Create the output directory if necessary
mkdir -p $out_dir

# Copy the bitstream and handoff to the specified directory
bitstream_out_path=$out_dir/$out_name.bit
handoff_out_path=$out_dir/$out_name.hwh
cp $bitstream_path $bitstream_out_path
cp $handoff_path $handoff_out_path

echo "Bitstream is copied: $bitstream_out_path"
echo "Handoff is copied: $handoff_out_path"

exit 0
