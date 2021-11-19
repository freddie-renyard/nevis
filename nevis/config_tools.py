import os
import json

class ConfigTools:

    @classmethod
    def run_fpga_config_wizard(cls):
        
        if not os.path.isfile("nevis/config/fpga_config.json"):
            print("[NeVIS]: FPGA configuration file does not exist. Running FPGA configuration wizard...")
            fpga_dict = {}

            # Obtain a valid FPGA name
            fpga_name = 0
            while type(fpga_name) != str:
                fpga_name = input("Enter name of FPGA development board: ")
            fpga_dict["board_name"] = fpga_name

            # Obtain a valid baud rate
            baud_rate = ""
            while True:
                baud_rate = input("Enter desired serial baud rate: ")
                if not baud_rate.isdecimal():
                    print("Invalid input: the input must be an integer; Try again.")
                else:
                    baud_rate = int(baud_rate)
                    break
            fpga_dict["baud_rate"] = baud_rate
                
            # Obtain a /dev/* serial port. TODO check how this works on non-POSIX machines
            while True:
                serial_port = input("Enter serial port in the form '/dev/*': ")
                if serial_port[0:5] != "/dev/":
                    print("Invalid input; Try again.")
                else:
                    break
            fpga_dict["serial_addr"] = serial_port
            
            with open("nevis/config/fpga_config.json", "w") as json_file:
                json.dump(fpga_dict, fp=json_file, indent=4)
            json_file.close()

        else:
            print("[NeVIS]: FPGA configuration file already exists. Proceeding...")
        return 1

    @staticmethod
    def load_data(filename):
        """Loads data from a file in the config directory.
        """
        filepath = "nevis/config/" + filename
        file = open(filepath)
        json_dict = json.load(file)
        file.close()
        return json_dict
            