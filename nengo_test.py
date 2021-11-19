import nengo
from nengo.network import Network
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np

# Define the input function to the neural population
def input_func(t):
    return np.sin(t *0.5* 2*np.pi)

# Define, build and run a simple NeVIS model
def run_simple_fpga_model():
    with model:

        input_node = nengo.Node(input_func)

        fpga_ens = NevisEnsembleNetwork(
            n_neurons=50,
            dimensions=1,
            compile_design=False
        )
        
        nengo.Connection(input_node, fpga_ens.input)
        #nengo.Connection(fpga_ens.output, fpga_ens.input)
        
        a = nengo.Ensemble(n_neurons=50, dimensions=1)
        nengo.Connection(input_node, a)

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



