from logging import Logger
import logging
import serial
import time
from nevis.config_tools import ConfigTools
from nevis.filetools import Filetools

logger = logging.getLogger("logs/nevis.log")

class FPGAPort:
    """An object which is used to handle serial communication to the target FPGA.
    Makes heavy use of the pyserial library

    Parameters
    ----------
    port : str
        The '/dev/*' location of the board in question. TODO this will need to be tested
        on a non-POSIX machine.
    baud : int
        The desired baud rate of the serial communication. This is compiled into the FPGA's
        design when the synthesis tools are run, and so is present in the config file.
    input_word_depth : [int]
        The bit depth of the input to the model. TODO develop for multiple dimensions.
    output_word_depth : [int]
        The bit depth of the output of the mode. TODO develop for multiple dimensions.
    output_scale : [int]
        The scale factor of the output word. TODO ensure that the transform argument is
        accounted for here.

    Attributes
    ----------
    port : str
        The '/dev/*' location of the board in question.
    baud : int
        The baud rate of the design.
    self.link_addr
        The serial link object used for data tx/rx.
    """
    def __init__(self):
        
        ConfigTools.run_fpga_config_wizard()
        
        # Open the json file with all the serial parameters
        serial_dict = ConfigTools.load_data("fpga_config.json")
        self.port = serial_dict["serial_addr"]
        self.baud_rate = serial_dict["baud_rate"]

        model_dict = ConfigTools.load_data("model_config.json")
        self.input_depths = model_dict["in_node_depths"]
        self.output_depths = model_dict["out_node_depths"]
        self.output_scales = model_dict["out_node_scales"]
        
        # Define an empty link address
        self.link_addr = 0

        self.begin_serial(5)

    def begin_serial(self, timeout):

        logger.info("INFO: Attempting to open serial port...")
        
        attempts = 0

        # Define the number of connection attempts
        rest_interval = 0.1
        max_attempts = timeout // rest_interval
        
        while type(self.link_addr) == int: 
            try:
                self.link_addr = serial.Serial(self.port, baudrate=self.baud_rate, timeout=0.0005)
                logger.info(('INFO: Opened serial port to device at' + self.link_addr.name))
            except:
                logger.info(('INFO: Connection failed. Re-attempting...attempts: '+ str(attempts)))
                attempts += 1
                time.sleep(rest_interval)
                pass

            if attempts >= max_attempts:
                logger.error(("ERROR: Serial connection to" + self.port + " failed."))
                print("[NeVIS]: ERROR: Serial connection to", self.port, " failed.")
                break

    def serial_comm_func(self):
        """ Function for sending and recieving data from the FPGA on each timestep.

        """
