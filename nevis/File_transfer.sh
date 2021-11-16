#!/bin/bash
# This script is run to transfer the compiled parameters to the Vivado host machine.
# It also executes the script which runs Vivado at the end of the script

echo "Beginning parameter file transfer to Vivado host machine..."
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH

rsync -a -v --stats --progress /Users/freddierenyard/Desktop/Compiler/proto_nevis/temp/ freddie@192.168.0.23:"/home/freddie/Desktop/Alchitry\\ Labs\\ Project\\ Things/Verilog\\ Projects/NEF_IMPL_PROTO/au_base_project.srcs/sources_1/new/"

cd ..

ssh freddie@192.168.0.23 'bash -s' < ./proto_nevis/run_vivado.sh