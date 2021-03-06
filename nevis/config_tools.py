import logging
from operator import mod
import os
import json

from numpy import ModuleDeprecationWarning

from nevis.filetools import Filetools

logger = logging.getLogger(__name__)

class ConfigTools:

    @classmethod
    def run_fpga_config_wizard(cls):
        
        if not os.path.isfile("nevis/config/fpga_config.json"):
            logger.error("[NeVIS]: FPGA configuration file does not exist. Running FPGA configuration wizard...")
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
            logger.error("[NeVIS]: FPGA configuration file already exists. Proceeding...")
            pass
            
    @classmethod
    def create_model_config_file(cls, in_node_depth, out_node_depth, out_node_scale, n_input_values, n_output_values):
        """ LEGACY
        Saves compiled model's interfacing parameters into a JSON config file.
        Parameters
        ----------
        in_node_depths : [int]
            A list of integers denoting the bit depths of the inputs.
        out_node_depths : [int]
            A list of integers denoting the bit depths of the outputs.
        out_node_scales : [int]
            A list of integers denoting the absolute scale factors of the outputs.
        n_values : int
            The number of values to decode - sum of the output dimensionalities of
            the ensembles represented.
        """
        
        model_dict = {}
        model_dict["in_node_depth"] = in_node_depth
        model_dict["out_node_depth"] = out_node_depth
        model_dict["out_node_scale"] = out_node_scale
        model_dict["n_input_values"] = n_input_values
        model_dict["n_output_values"] = n_output_values

        with open("nevis/config/model_config.json", "w") as json_file:
            json.dump(model_dict, fp=json_file, indent=4)
        json_file.close()

    @classmethod
    def create_full_model_config(
        cls, 
        in_node_depth, in_node_dims, in_node_radix,
        out_node_depth, out_node_dims, out_node_scale
        ):
        """Creates a JSON file which describes the parameters needed
        to interpret the binary data from the UART.

        
        Parameters
        ----------
        node_depth: int
            The bit depth of the i/o values
        node_dims: lst
            The dimensionalities of each of the i/o nodes. Also implicitly contains
            the number of i/o nodes.
        out_node_scale: int
            The number of binary orders of magnitude to divide by
            to achieve the correct output scale
        """

        model_dict = {}

        model_dict["in_node_depth"] = in_node_depth
        model_dict["in_node_dims"]  = in_node_dims
        model_dict["in_node_radix"] = in_node_radix

        model_dict["out_node_depth"] = out_node_depth
        model_dict["out_node_dims"]  = out_node_dims
        model_dict["out_node_scale"] = out_node_scale

        with open("nevis/config/model_config.json", "w") as json_file:
            json.dump(model_dict, fp=json_file, indent=4)
        json_file.close()

    @staticmethod
    def load_data(filename):
        """Loads data from a file in the config directory.
        """
        filepath = "nevis/config/" + filename
        file = open(filepath)
        json_dict = json.load(file)
        file.close()
        return json_dict
            
    @staticmethod
    def purge_model_config():
        filepath = "nevis/config/model_config.json"
        if os.path.isfile(filepath):
            os.remove(filepath)