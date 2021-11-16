#!/bin/bash
# This script is run to transfer the compiled parameters to the Vivado host machine.
# It also executes the script which runs Vivado at the end of the script

echo "Beginning parameter file transfer to Vivado host machine..."

# The path to the Vivado build files on the host machine
HOST_PATH="/home/freddie/Desktop/Alchitry\ Labs\ Project\ Things/Verilog\ Projects/NEF_IMPL_PROTO/au_base_project.srcs/sources_1/new/"

# Absolute path this script is in. Will not work for symlinks.
SCRIPTPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CACHEPATH="/file_cache/"
FINALPATH="$SCRIPTPATH$CACHEPATH"

# Transfer the files over SSH to the host machine
rsync -a -v --stats --progress $FINALPATH freddie@192.168.0.23:HOST_PATH

# Run the Vivado execution script on the host machine
NEXTSCRIPTPATH="${SCRIPTPATH}/run_vivado.sh"
ssh freddie@192.168.0.23 'bash -s' < $NEXTSCRIPTPATH