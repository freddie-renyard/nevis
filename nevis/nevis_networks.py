import nengo

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
    label : str, optional (Default: None)
        A descriptive label for the connection.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this network will be added to the current container. If
        ``None``, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.
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
        The connection object used to configure the learning connection
        implementation on the FPGA board.
    feedback : `nengo.Connection`
        The connection object used to configure the recurrent connection
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
        label=None,
        seed=None,
        add_to_container=None,
    ):
        print("This method is under construction....")
        
        # Create serial object
        # TODO

        self.input_dimensions = dimensions
        if dimensions != 1:
            print("ERROR: Current implementation of NeVIS does not support dimensions higher than 1.")
            # TODO Raise error apprpriately.

        # Set the output dimensions - currently only 1D is supported.
        self.output_dimensions = 1
        
        self.neuron_type = nengo.neurons.LIF()

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
                eval_poins  = eval_points
            )

    def extract_params():
        print("This method is under construction...")

    
