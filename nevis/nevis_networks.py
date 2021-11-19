from functools import partial
import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import Copy, Reset, SimPyFunc
import numpy as np
import math
from nevis import neuron_classes
from nevis import serial_link
#import neuron_classes

class NevisEnsembleNetwork(nengo.Network):

    """ The code below is derived from the interfacing implementation found in the existing NengoFPGA backend.
    Source: https://github.com/nengo/nengo-fpga/blob/master/nengo_fpga/networks/fpga_pes_ensemble_network.py 
    
    An ensemble to be run on the FPGA. 
    Parameters
    ----------
    fpga_serial_addr : str
        The serial port of the FPGA, in the format "/dev/tty.<address here>"
    n_neurons : int
        The number of neurons.
    dimensions : int
        The number of representational dimensions. Under current
        implementation, 1 is the only valid input.
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
    tau : int or float
        The time constant of the synapses in the model.
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
    """

    def __init__(
        self,
        fpga_serial_addr,
        n_neurons,
        dimensions,
        function=nengo.Default,
        transform=nengo.Default,
        eval_points=nengo.Default,
        timeout=2,
        compile_design=True,
        label=None,
        seed=None,
        add_to_container=None,
        t_pstc=0.1275
    ):
        
        # TODO Check compile_design param
        # TODO Check whether the network params are the same as those passed to the object,
        #   else recompile and synthesise the design. Do this via the JSON config file that is
        #   yet to exist.

        self.input_dimensions = dimensions
        if dimensions != 1:
            print("ERROR: Current implementation of NeVIS does not support dimensions higher than 1.")
            # TODO Raise error apprpriately.

        # Set the output dimensions - currently only 1D is supported.
        self.output_dimensions = 1
        
        self.neuron_type = nengo.neurons.LIF()
        self.t_pstc = t_pstc

        self.seed = seed

        super().__init__(label, seed, add_to_container)

        self.serial_addr = fpga_serial_addr
        self.timeout = timeout

        # Set up a dummy model with the input and output Nodes used to interface
        # with Nengo. This approach is taken from the source listed above.
        with self:
            self.input = nengo.Node(size_in=self.input_dimensions, label="input")
            self.output = nengo.Node(size_in=self.output_dimensions, label="output")

            self.ensemble = nengo.Ensemble(
                n_neurons   = n_neurons,
                dimensions  = self.input_dimensions,
                neuron_type = nengo.neurons.LIF(),
                eval_points = eval_points,
                label       = "Dummy Ensemble"
            )
            nengo.Connection(self.input, self.ensemble, synapse=None)

            self.connection = nengo.Connection(
                self.ensemble, # Pre object
                self.output,   # Post object
                function    = function,
                transform   = transform,
                eval_points  = eval_points
            )

def compile_and_save_params(model, network):
    """ Extracts the parameters from the network, compiles them into
    the appropriate format for the NeVIS hardware, saves them to the 
    file cache, and invokes the method which transfers and executes
    Vivado on a remote server.

    Again, implementation takes details from the existing backend.
    """
    
    # Generate the model which the parameters will be taken from
    param_model = nengo.builder.Model(dt=model.dt)
    nengo.builder.network.build_network(param_model, network)

    # Gather simulation parameters - identical across all ensembles
    sim_args = {}
    sim_args["dt"] = model.dt

    # Gather ensemble parameters - vary between ensembles
    ens_args = {}
    ens_args["n_neurons"] = network.ensemble.n_neurons
    ens_args["input_dimensions"] = network.input_dimensions
    ens_args["output_dimensions"] = network.output_dimensions
    ens_args["bias"] = param_model.params[network.ensemble].bias
    ens_args["t_rc"] = network.ensemble.neuron_type.tau_rc / sim_args["dt"]

    # scaled_encoders = gain * encoders
    # TODO this is computationally wasteful, but the way that the Encoder 
    # object is designed at present makes the code below the most readable 
    # solution. Change the Encoder so that this is not the case.
    ens_args["encoders"] = param_model.params[network.ensemble].encoders
    ens_args["gain"] = param_model.params[network.ensemble].gain

    # Gather refractory period
    ens_args["ref_period"] = network.ensemble.neuron_type.tau_ref / sim_args["dt"]

    # Tool for painlessly investigating the parameters of Nengo objects
    l = dir(param_model.params[network.connection])
    print(l)

    print(type(ens_args["t_rc"]))

    conn_args = {}
    conn_args["weights"] = param_model.params[network.connection].weights
    conn_args["t_pstc"] = network.t_pstc / sim_args["dt"]
    conn_args["pstc_scale"] = 1.0 - math.exp(-1 / network.t_pstc) # Timestep has been normalised to 1

    # Define the compiler params. TODO write an optimiser function to
    # define these params automatically.
    comp_args = {}
    comp_args["radix_encoder"] = 8
    comp_args["bits_input"] = 8
    comp_args["radix_input"] = comp_args["bits_input"] - 1
    comp_args["radix_weights"] = 6
    comp_args["n_dv_post"] = 10
    comp_args["n_activ_extra"] = 6
    comp_args["min_float_val"] = 1*2**-20

    # TODO SCALE ALL OF THE TEMPORAL PARAMS BY DT

    # Compile an ensemble (NeVIS - Encoder) TODO ensure that this distinction is correct
    input_hardware = neuron_classes.Encoder_Floating(
        n_neurons=ens_args["n_neurons"],
        gain_list=ens_args["gain"],
        encoder_list=ens_args["encoders"],
        bias_list=ens_args["bias"],
        t_rc=ens_args["t_rc"],
        n_x=comp_args["bits_input"],
        radix_x=comp_args["radix_input"],
        radix_g=comp_args["radix_encoder"],
        radix_b=comp_args["radix_encoder"],
        n_dv_post=comp_args["n_dv_post"],
        verbose=False
    )

    # Compile an output node (Nevis - Synapses)
    # TODO Check whether weights are prescaled with pstc_scale value (if not default to one)
    output_hardware = neuron_classes.Synapses_Floating(
        n_neurons=ens_args["n_neurons"],
        pstc_scale=conn_args["pstc_scale"],
        decoders_list=conn_args["weights"][0], # Take the zeroeth dimension of this array as it is
        # a dot product of the decoders and the encoders of the next section (as this dot product
        # is used for hardware memory optimisation)
        encoders_list=[1], # Indicates a positive weight addition
        n_activ_extra=comp_args["n_activ_extra"],
        radix_w=comp_args["radix_weights"],
        minimum_val=comp_args["min_float_val"],
        verbose=True
    )

@nengo.builder.Builder.register(NevisEnsembleNetwork)
def build_NevisEnsembleNetwork(model, network):

    # TODO Perform hardware checks before preceding with FPGA build.

    # Instantiate a serial link object
    serial_port = serial_link.FPGAPort()

    # Extract relevant params
    compile_and_save_params(model, network)

    # Define input signal and assign it to the model's input
    input_sig = Signal(np.zeros(network.input_dimensions), name="input")
    model.sig[network.input]["in"] = input_sig
    model.sig[network.input]["out"] = input_sig
    model.add_op(Reset(input_sig))

    # Build the input signal into the model
    input_sig = model.build(nengo.synapses.Lowpass(network.t_pstc), input_sig)

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
            fn=partial(serial_link.serial_comm_func, net=network, dt=model.dt),
            t=model.time,
            x=serial_port_input_sig
        )
    )