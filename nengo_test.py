import nengo
from nengo.network import Network
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np

# Define the input function to the neural population
def input_func(t):
    return np.sin(t * 2*np.pi)

# Define, build and run a simple NeVIS model
def run_simple_fpga_model():
    with model:

        input_node = nengo.Node(input_func)

        fpga_ens = NevisEnsembleNetwork(
            n_neurons=50,
            dimensions=2,
            compile_design=True
        )

        nengo.Connection(input_node, fpga_ens.input)

        # Create an output node to display the output value of the 
        # FPGA ensemble.
        output_node = nengo.Node(
            size_in=1, 
            size_out=1,
            label="output_node"
        )

        nengo.Connection(fpga_ens.output, output_node)

# Define, build and run a default Nengo network
def run_default_model():
    with model:
        stim = nengo.Node([0])
        a = nengo.Ensemble(n_neurons=50, dimensions=1)
        nengo.Connection(stim, a)

model = nengo.Network()

run_simple_fpga_model()

import nengo_gui
nengo_gui.GUI(__file__).start()



