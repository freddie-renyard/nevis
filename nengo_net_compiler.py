from cProfile import label
import nengo
from nengo.network import Network
from numpy.core.numeric import True_
from nevis.global_tools import Global_Tools
from nevis.network_compiler import NevisCompiler
import numpy as np

from nevis.nevis_networks import NevisNetwork

# Define the input function to the neural population
def input_func_sin(t):
    return np.sin(t * 2*np.pi) #, np.cos(t * 2*np.pi)

def input_func_cos(t):
    return np.cos(t * 2 * np.pi)

def target_function(x):
    return x

model = Network()

dimensions = 1

with model:

    nevis_model = NevisNetwork(compile_network=True)
    with nevis_model:
        
        t_pstc = Global_Tools.inverse_pstc(128, 0.001)
        tau_rc = Global_Tools.inverse_rc(8, 0.001)

        in_node_1 = nengo.Node(size_in=dimensions)
        #in_node_2 = nengo.Node(size_in=dimensions)

        a = nengo.Ensemble(
            n_neurons=50, 
            dimensions=dimensions,
            neuron_type=nengo.neurons.LIF(tau_rc)
        )

        """
        b = nengo.Ensemble(
            n_neurons=50, 
            dimensions=dimensions,
            neuron_type=nengo.neurons.LIF(tau_rc)
        )
        """

        nengo.Connection(in_node_1, a)
        #nengo.Connection(in_node_2, b)
        
        output_node_1 = nengo.Node(size_in=dimensions)
        nengo.Connection(a, output_node_1)

        #output_node_2 = nengo.Node(size_in=dimensions)
        #nengo.Connection(b, output_node_2)

    stim1 = nengo.Node(input_func_sin)
    nengo.Connection(stim1, nevis_model.nodes[0])

    #stim2 = nengo.Node(input_func_cos)
    #nengo.Connection(stim2, nevis_model.nodes[1])

import nengo_gui
nengo_gui.GUI(__file__).start()