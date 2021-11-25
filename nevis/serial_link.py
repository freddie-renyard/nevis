from logging import Logger
import logging
import serial
import time
from nevis.config_tools import ConfigTools
from nevis.filetools import Filetools
import math
import bitstring

logger = logging.getLogger("logs/nevis.log")

class FPGAPort:
    """An object which is used to handle serial communication to the target FPGA.
    Makes heavy use of the pyserial library

    Parameters
    ----------
    timeout : int
        The number of seconds to wait for a board connection.

    Attributes
    ----------
    port : str
        The '/dev/*' location of the board in question.
    baud : int
        The baud rate of the design.
    self.link_addr
        The serial link object used for data tx/rx.
    """
    def __init__(self, timeout):
        
        ConfigTools.run_fpga_config_wizard()
        
        # Open the json file with all the serial parameters
        serial_dict = ConfigTools.load_data("fpga_config.json")
        self.port = serial_dict["serial_addr"]
        self.baud_rate = serial_dict["baud_rate"]

        model_dict = ConfigTools.load_data("model_config.json")
        self.input_depths = model_dict["in_node_depths"]
        self.output_depths = int(model_dict["out_node_depths"][0])
        self.output_scales = model_dict["out_node_scales"][0]
        self.n_values = model_dict["n_values"]

        self.bytes_to_read = math.ceil((self.output_depths*self.n_values) / 8.0)
        
        # Define an empty link address
        self.link_addr = 0

        self.begin_serial(timeout)

    def begin_serial(self, timeout):

        logger.info("INFO: Attempting to open serial port...")
        print("[NeVIS]: Attempting to open serial port...")
        
        attempts = 0

        # Define the number of connection attempts
        rest_interval = 0.1
        max_attempts = timeout // rest_interval
        
        while type(self.link_addr) == int: 
            try:
                self.link_addr = serial.Serial(self.port, baudrate=self.baud_rate, timeout=0.0005) # , timeout=0.0005
                logger.info(('INFO: Opened serial port to device at' + self.link_addr.name))
                print('[NeVIS]: Opened serial port to device at', self.link_addr.name)
            except:
                if max_attempts != 0:
                    logger.info(('INFO: Connection failed. Re-attempting...attempts: '+ str(attempts)))
                attempts += 1
                time.sleep(rest_interval)

            if attempts >= max_attempts and max_attempts != 0:
                logger.error(("ERROR: Serial connection to" + self.port + " failed."))
                print("[NeVIS]: ERROR: Serial connection to", self.port, " failed.")
                break

    def serial_comm_func(self, t, d, net, dt):
        """ Function for sending and recieving data from the FPGA on each timestep.

        """
        
        # Scale the input value up
        in_scale = 2 ** (self.input_depths[0] - 1)
        output_num = int(d[0] * in_scale)
        in_x = self.twos_complementer(output_num)
        
        try:
            self.link_addr.write(in_x)
        except:
            pass

        rx_data = self.link_addr.read(size=self.bytes_to_read)
        
        #ser.flush()
        hardware_vals = [0 for _ in range(self.n_values)]
        
        #if len(rx_data) == self.bytes_to_read:
        data = bitstring.BitArray(rx_data).bin
        for i in range(self.n_values):
            bytes_obj = bitstring.Bits(bin=data[self.output_depths*i:self.output_depths*(i+1)]).tobytes()
            hardware_vals[i] = int.from_bytes(bytes_obj, byteorder="big", signed=True) / (2 ** self.output_scales * 2**4)
            
        return hardware_vals

    def twos_complementer(self, value):
        # TODO Heavily optimise this.
        
        sign = (value < 0)
        value = abs(value)

        if sign:
            value = bin(value)
            value = value[2:]

            value = value.replace('1', '2')
            value = value.replace('0', '1')
            value = value.replace('2', '0')

            value = ('1')*(8 - len(value)) + value
            
            value = int(value, 2)
        
        return bytes([value])
        
