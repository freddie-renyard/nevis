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
    x = [np.sin(t * 2*np.pi + x*0.5) for x in range(dimensions)]
    return x

def input_func_cos(t):
    x = [np.cos(t * 2*np.pi + x*0.5) for x in range(dimensions)]
    return x

def target_function(x):
    return x**2

model = Network()

dimensions = 1

with model:

    nevis_model = NevisNetwork(compile_network=True)
    with nevis_model:
        
        t_pstc = Global_Tools.inverse_pstc(128, 0.001)
        tau_rc = Global_Tools.inverse_rc(8, 0.001)

        in_node_1 = nengo.Node(size_in=dimensions)

        a = nengo.Ensemble(
            n_neurons=100, 
            dimensions=dimensions,
            neuron_type=nengo.neurons.LIF(tau_rc),
            radius=1
        )
        
        nengo.Connection(in_node_1, a)
        
        output_node_1 = nengo.Node(size_in=dimensions)

        """
        b = nengo.Ensemble(
            n_neurons=100, 
            dimensions=dimensions,
            neuron_type=nengo.neurons.LIF(tau_rc),
            radius=1
        )
        """

        #nengo.Connection(a, b, synapse=t_pstc)
        #nengo.Connection(b, output_node_1, synapse=t_pstc)

        nengo.Connection(a, output_node_1, synapse=t_pstc)

    stim1 = nengo.Node(input_func_sin)
    nengo.Connection(stim1, nevis_model.nodes[0])

    a = nengo.Ensemble(
        n_neurons=100, 
        dimensions=dimensions,
        neuron_type=nengo.neurons.LIF(tau_rc),
        radius=1
    )

    output_node = nengo.Node(size_in=dimensions)

    nengo.Connection(stim1, a)
    nengo.Connection(a, output_node, synapse=t_pstc)
    
import nengo_gui
nengo_gui.GUI(__file__).start()