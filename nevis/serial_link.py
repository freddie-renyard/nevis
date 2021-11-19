import serial
import time

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
    def __init__(self, port, baud, input_word_depth, output_word_depth, output_scale):
        print("FPGA Port says hello world...")
        self.port = port
        self.baud_rate = baud

        # Define an empty link address
        self.link_addr = 0
        pass

    def begin_serial(self, timeout):
        print("[NeVIS]: Attempting to open serial port...")
        
        attempts = 0

        # Define the number of connection attempts
        rest_interval = 0.1
        max_attempts = timeout // rest_interval
        
        while type(self.link_addr) == int: 
            try:
                self.link_addr = serial.Serial(self.port, baudrate=self.baud_rate, timeout=0.0005)
                print('[NeVIS]: Opened serial port to device at ', self.link_addr.name)
            except:
                time.sleep(rest_interval)
                pass

            if attempts >= max_attempts:
                print("[NeVIS]: Serial connection to", self.port, "failed.")
                break

    def serial_comm_func(self):
        """ Function for sending and recieving data from the FPGA on each timestep.

        """
