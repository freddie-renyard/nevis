#!/bin/bash
# This script is run when the remote Vivado host is logged into over SSH.

echo "Attempting Vivado run..."
source /home/freddie/Applications/Vivado/2020.2/settings64.sh vivado
vivado -mode tcl

open_project {/home/freddie/Desktop/Alchitry Labs Project Things/Verilog Projects/NEF_IMPL_PROTO/au_base_project.xpr}
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
wait_on_run write_bitstream
exit
echo "Finished!!"