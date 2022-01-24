from logging import Logger
import logging
import serial
import time
from nevis.config_tools import ConfigTools
from nevis.filetools import Filetools
import math
import bitstring

from timeit import default_timer as timer

logger = logging.getLogger("logs/nevis.log")

class FPGAPort:
    """ LEGACY - TO BE DELETED AFTER FULL NETWORK IMPLEMENTATION IS REALISED
    An object which is used to handle serial communication to the target FPGA.
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

        self.input_depth = model_dict["in_node_depth"]
        self.n_input_values = model_dict["n_input_values"]
        self.bytes_to_send = math.ceil((self.input_depth * self.n_input_values) / 8.0)

        self.output_depths = int(model_dict["out_node_depth"])
        self.output_scales = model_dict["out_node_scale"]
        self.n_values = model_dict["n_output_values"]
        self.bytes_to_read = math.ceil((self.output_depths*self.n_values) / 8.0)
        self.rx_bits = self.n_values * (self.output_depths)
        self.bit_excess = self.bytes_to_read * 8 - self.rx_bits
        
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
        in_scale = 2 ** (self.input_depth - 1)

        print(d)

        full_tx_word = ""
        d = d[0:int(len(d)/2)]
        for value in d:
            
            output_num = int(value * in_scale)
            bit_obj = bitstring.Bits(int=output_num, length=self.input_depth)
            full_tx_word += bit_obj.bin #  bit_obj.bin + full_tx_word
        in_total = bitstring.Bits(bin=full_tx_word)
        #print(in_total.bin, len(in_total.bytes))

        t_start = timer()

        try:
            self.link_addr.write(in_total.bytes)
        except:
            pass

        rx_data = self.link_addr.read(size=self.bytes_to_read)

        t_end = timer()

        #print("Microsecs: {}".format((t_end - t_start)*10**6))
        #input()

        #ser.flush()
        hardware_vals = [0 for _ in range(self.n_values)]
    
        data = bitstring.BitArray(rx_data).bin
        #print(data)
        # Trim the excess bits generated by the the UART, which only transmits bytes
        data = data[self.bit_excess:]
        #print(data)

        if len(data) == self.rx_bits:
            for i in range(self.n_values):
                
                bytes_obj = bitstring.BitArray(bin=data[(self.n_values-i-1)*self.output_depths:self.output_depths*(self.n_values-i)])
                bytes_obj = bytes_obj.int
                hardware_vals[i] = bytes_obj / (2 ** (self.output_scales))
                
        return hardware_vals

    def decode_binary(self, bin_str):
        if bin_str[-1] == "1":
            bin_str = bin_str.replace('1', '2')
            bin_str = bin_str.replace('0', '1')
            bin_str = bin_str.replace('2', '0')

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
        
class FPGANetworkPort:

    def __init__(self, timeout):
        
        ConfigTools.run_fpga_config_wizard()
        
        # Open the json file with all the serial parameters
        serial_dict = ConfigTools.load_data("fpga_config.json")
        self.port = serial_dict["serial_addr"]
        self.baud_rate = serial_dict["baud_rate"]

        model_dict = ConfigTools.load_data("model_config.json")

        # Input side
        self.in_node_depth = model_dict["in_node_depth"]
        self.in_node_dims = model_dict["in_node_dims"]
        self.in_node_num = len(self.in_node_dims)

        self.bytes_to_send = math.ceil((self.in_node_depth * sum(self.in_node_dims) * self.in_node_num) / 8.0)

        # Output side
        self.out_node_depth = int(model_dict["out_node_depth"])
        self.out_node_scale = model_dict["out_node_scale"]
        self.out_node_dims = model_dict["out_node_dims"]
        self.out_node_num = len(self.out_node_dims)

        self.rx_bits = self.out_node_depth * sum(self.out_node_dims) * self.out_node_num
        self.bytes_to_read = math.ceil(self.rx_bits / 8.0)
        self.bit_excess = self.bytes_to_read * 8 - self.rx_bits
        
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
        in_scale = 2 ** (self.in_node_depth - 1)

        full_tx_word = ""
        d = d[0:int(len(d)/2)]
        for value in d:
            
            output_num = int(value * in_scale)
            bit_obj = bitstring.Bits(int=output_num, length=self.in_node_depth)
            full_tx_word += bit_obj.bin #  bit_obj.bin + full_tx_word
        in_total = bitstring.Bits(bin=full_tx_word)
        #print(in_total.bin, len(in_total.bytes))

        t_start = timer()

        try:
            self.link_addr.write(in_total.bytes)
        except:
            pass

        rx_data = self.link_addr.read(size=self.bytes_to_read)

        t_end = timer()

        #input()

        #ser.flush()
        hardware_vals = [0 for _ in range(self.out_node_num)]
    
        data = bitstring.BitArray(rx_data).bin
        
        # Trim the excess bits generated by the the UART, which only transmits bytes
        data = data[self.bit_excess:]
        print(data)

        if len(data) == self.rx_bits:
            for i in range(self.out_node_num):
                
                bytes_obj = bitstring.BitArray(bin=data[(self.out_node_num-i-1)*self.out_node_depth:self.out_node_depth*(self.out_node_num-i)])
                bytes_obj = bytes_obj.int
                hardware_vals[i] = bytes_obj / (2 ** (self.out_node_scale))

        test_out = [0.2, 0.3, 0.4, -0.9, -0.8, -0.7]

        return hardware_vals