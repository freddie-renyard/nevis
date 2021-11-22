# A setup script, which creates appropriate config files needed for NeVIS.

from nevis.nevis import config_tools
import os

print("Running setup script...")
print("/********* NeVIS **********/")

print("Welcome to NeVIS setup! FPGA and server setup is required.")
print("Creating config directory...")
if os.path.isdir("nevis/config"):
    os.mkdir("nevis/config")
print("Directory created.")

# Call the FPGA setup script.

# Call the server setup script.

# Install all the prerequisites.