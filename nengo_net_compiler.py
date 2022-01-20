import nengo
from nengo.network import Network
from numpy.core.numeric import True_
from nevis.global_tools import Global_Tools
from nevis.network_compiler import NevisCompiler
import numpy as np

from nevis.nevis_networks import NevisNetwork

# Define the input function to the neural population
def input_func(t):
    return 1 #np.sin(t * 2*np.pi)

def target_function(x):
    return x

model = Network()

with model:

    stim = nengo.Node(input_func)

    t_pstc = Global_Tools.inverse_pstc(128, 0.001)
    tau_rc = Global_Tools.inverse_rc(8, 0.001)

    a = nengo.Ensemble(
        n_neurons=50, 
        dimensions=1,
        neuron_type=nengo.neurons.LIF(tau_rc)
    )

    nengo.Connection(stim, a)
    output_node_2 = nengo.Node(size_in=1)
    nengo.Connection(a, output_node_2)

NevisCompiler().compile_network(model)

import nengo_gui
nengo_gui.GUI(__file__).start()