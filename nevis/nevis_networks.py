from functools import partial
import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import Copy, Reset, SimPyFunc
import numpy as np
import math
import logging

from numpy.lib import index_tricks
from nevis import neuron_classes
from nevis import serial_link
from nevis.config_tools import ConfigTools
from nevis.filetools import Filetools
from nevis.memory_compiler import Compiler
import os
import sys
from subprocess import check_call
from nevis.global_tools import Global_Tools
from nevis.network_compiler import NevisCompiler

from nevis.serial_link import FPGAPort

logging.basicConfig(
    level=logging.INFO,
    filename="nevis/logs/logs.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

class NevisEnsembleNetwork(nengo.Network):

    """ The code below is derived from the interfacing implementation found in the existing NengoFPGA backend.
    Source: https://github.com/nengo/nengo-fpga/blob/master/nengo_fpga/networks/fpga_pes_ensemble_network.py 
    
    An ensemble to be run on the FPGA. 
    Parameters
    ----------
    n_neurons : int
        The number of neurons.
    dimensions : int
        The number of representational dimensions.
    tau_rc : float
        The time constant of the LIF membrane.
    function : callable or 
               optional (Default: None)
        Function to compute across the connection. 
    transform : (size_out, size_mid) array_like, optional \
                (Default: ``np.array(1.0)``)
        Linear transform mapping the pre output to the post input.
        This transform is in terms of the sliced size; if either pre
        or post is a slice, the transform must be shaped according to
        the sliced dimensionality. Additionally, the function is applied
        before the transform, so if a function is computed across the
        connection, the transform must be of shape ``(size_out, size_mid)``.
    eval_points : (n_eval_points, size_in) array_like or int, optional \
                  (Default: None)
        Points at which to evaluate ``function`` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
        If None, will use the eval_points associated with ``pre``.
    timeout : float or int, optional
        Number of seconds to wait for FPGA connection over UART
    compile_design : bool
        Whether to recompile and synthesise the given network. 
    label : str, optional (Default: None)
        A descriptive label for the connection.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this network will be added to the current container. If
        ``None``, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.
    t_pstc : int or float
        The time constant of the synapses in the model.
    verbose : bool
        Whether to print the logs to the commadn line
    Attributes
    ----------
    input : `nengo.Node`
        A node that serves as the input interface between external Nengo
        objects and the FPGA board.
    output : `nengo.Node`
        A node that serves as the output interface between the FPGA board and
        external Nengo objects.
    ensemble : `nengo.Ensemble`
        An ensemble object whose parameters are used to configure the
        ensemble implementation on the FPGA board.
    connection : `nengo.Connection`
        The connection object used to configure the output connection
        implementation on the FPGA board.
    compile_design : bool
        Whether to compile the model from scratch.
    """

    def __init__(
        self,
        n_neurons,
        dimensions,
        tau_rc=0.016,
        function=nengo.Default,
        transform=nengo.Default,
        eval_points=nengo.Default,
        timeout=2,
        compile_design=True,
        label=None,
        seed=30,
        add_to_container=None,
        t_pstc=0.1275,
        verbose=False
    ):
        
        # TODO Check compile_design param
        # TODO Check whether the network params are the same as those passed to the object,
        #   else recompile and synthesise the design. Do this via the JSON config file that is
        #   yet to exist.
        # Add n_neurons, dimensions etc. to the model_config.json file. Compare these attributes on each
        # new object instantiation, and if any don't match, re-run the compilation process.

        self.compile_design = compile_design

        if verbose:
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        #ConfigTools.run_fpga_config_wizard()

        self.input_dimensions = dimensions
        self.output_dimensions = self.get_output_dims(dimensions, function)
        
        self.t_pstc = t_pstc
        self.tau_rc = tau_rc
        self.neuron_type = nengo.neurons.LIF(self.tau_rc)

        self.seed = seed

        super().__init__(label, seed, add_to_container)

        self.timeout = timeout

        # Set up a dummy model with the input and output Nodes used to interface
        # with Nengo. This approach is taken from the source listed above.
        with self:
            self.input = nengo.Node(size_in=self.input_dimensions, label="input")
            self.output = nengo.Node(size_in=self.output_dimensions, label="output")

            self.ensemble = nengo.Ensemble(
                n_neurons   = n_neurons,
                dimensions  = self.input_dimensions,
                neuron_type = self.neuron_type,
                eval_points = eval_points,
                label       = " ",
                seed        = self.seed,
                radius      = 1
            )
            nengo.Connection(self.input, self.ensemble, synapse=t_pstc)
            
            self.connection = nengo.Connection(
                self.ensemble, # Pre object
                self.output,   # Post object
                function    = function,
                transform   = transform,
                eval_points  = eval_points
            )

    def get_output_dims(self, dimensions, function):
        """ Check if your current architecture supports
        array-like inputs for functions.
        """
        if callable(function):
            return len(function(np.zeros(dimensions)))
        elif function is nengo.Default:
            return dimensions
        else:
            logger.error("Only callable functions are currently supported.")
            raise nengo.exceptions.ValidationError("Only callable functions are currently supported.", "function", self)

def call_synthesis_server():
    """Call the script that transfers the compiled files to
    the Vivado server machine.
    """
    cwd = os.getcwd()

    server_config = ConfigTools.load_data("server_config.json")

    server_path = server_config["project_dir"]
    server_addr = server_config["ssh_addr"]
    vivado_loc = server_config["vivado_loc"]
    project_path = server_config["project_loc"]
    script_path = cwd + "/nevis/File_transfer.sh %s %s %s %s"
    check_call(script_path % (server_path, server_addr, vivado_loc, project_path), shell=True)

def compile_and_save_params(model, network):
    """ Extracts the parameters from the network, compiles them into
    the appropriate format for the NeVIS hardware, saves them to the 
    file cache, and invokes the method which transfers and executes
    Vivado on a remote server.

    Again, implementation takes details from the existing backend.
    """
    if network.compile_design:

        # Purge previous build caches 
        ConfigTools.purge_model_config()
        Filetools.purge_directory("nevis/file_cache")

        # Generate the model which the parameters will be taken from
        param_model = nengo.builder.Model(dt=model.dt)
        nengo.builder.network.build_network(param_model, network)

        # Compile the network and save params to .mem files.
        compiler = NevisCompiler().compile_ensemble(
            model   = param_model, 
            network = network
        )

        call_synthesis_server()

@nengo.builder.Builder.register(NevisEnsembleNetwork)
def build_NevisEnsembleNetwork(model, network):

    # TODO Perform hardware checks before preceding with FPGA build.
    
    # Extract relevant params
    compile_and_save_params(model, network)

    # Instantiate a serial link object - no timeout if the model is being compiled.
    if network.compile_design:
        timeout=0
    else:
        timeout=10
    
    serial_port = serial_link.FPGAPort(timeout)

    # Define input signal and assign it to the model's input
    input_sig = Signal(np.zeros(network.input_dimensions), name="input")
    model.sig[network.input]["in"] = input_sig
    model.sig[network.input]["out"] = input_sig
    model.add_op(Reset(input_sig))

    # Build the input signal into the model
    input_sig = model.build(nengo.synapses.Lowpass(0), input_sig)

    # Define the output signal
    output_sig = Signal(np.zeros(network.output_dimensions), name="output")
    model.sig[network.output]["out"] = output_sig

    # Build the output connection into the model
    if network.connection.synapse is not None:
        model.build(network.connection.synapse, output_sig)

    # Combine the data ports to allow for the serial interface to 
    # communicate with Nengo
    serial_port_input_sig = Signal(
        np.zeros(network.input_dimensions + network.output_dimensions),
        name="serial_port_input"
    )

    # Copy the tx data to the input of the model in Nengo
    model.add_op(
        Copy(
            input_sig,
            serial_port_input_sig,
            dst_slice=slice(0, network.input_dimensions)
        )
    )

    model.add_op(  
        SimPyFunc(
            output=output_sig,
            fn=partial(serial_port.serial_comm_func, net=network, dt=model.dt),
            t=model.time,
            x=serial_port_input_sig
        )
    )