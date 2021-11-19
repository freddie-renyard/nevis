import os
import json

class ConfigTools:

    def run_fpga_config_wizard():

        if not os.path.isfile("nevis/config/fpga_config.json"):
            print("[NeVIS]: FPGA configuration file does not exist. Running FPGA configuration wizard...")
            fpga_dict = {}

            # TODO add error checking to these steps.

            # Obtain a valid FPGA name
            fpga_name = 0
            while type(fpga_name) != str:
                fpga_name = input("Enter name of FPGA development board: ")
            fpga_dict["board_name"] = fpga_name

            # Obtain a valid baud rate
            baud_rate = ""
            while type(baud_rate) != int:
                baud_rate = int(input("Enter desired serial baud rate: "))
            fpga_dict["baud_rate"] = baud_rate
                
            # Obtain a /dev/* serial port. TODO check how this works on non-POSIX machines.
            serial_port = 0
            while type(serial_port) != str:
                serial_port = input("Enter serial port in the form '/dev/*': ")
            fpga_dict["serial_addr"] = serial_port
            
            with open("nevis/config/fpga_config.json", "w") as json_file:
                json.dump(fpga_dict, fp=json_file, indent=4)
        else:
            print("[NeVIS]: FPGA configuration file already exists. Proceeding...")