import nengo
from nengo.network import Network
from numpy.core.numeric import True_
from nevis.global_tools import Global_Tools
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np

# Define the input function to the neural population
def input_func(t):
    return np.sin(t *0.5* 2*np.pi) #, np.cos(t *0.5* 2*np.pi), np.sin(t *0.25* 2*np.pi), np.cos(t *0.25* 2*np.pi)

def target_function(x):
    """
    return_vals = []
    for i in range(20):
        return_vals.append(x**i)
    return return_vals
    """
    return x

# Define, build and run a simple NeVIS model
model = nengo.Network()

with model:

    input_node = nengo.Node(input_func)

    t_pstc = Global_Tools.inverse_pstc(128, 0.001)
    tau_rc = Global_Tools.inverse_rc(8, 0.001)

    neurons = 50
    dimensions = 4

    fpga_ens = NevisEnsembleNetwork(
        n_neurons=neurons,
        dimensions=dimensions,
        compile_design=False,
        t_pstc=t_pstc,
        tau_rc=tau_rc,
        function=target_function
    )
    nengo.Connection(input_node, fpga_ens.input)
    
    a = nengo.Ensemble(
        n_neurons=neurons, 
        dimensions=dimensions, 
        neuron_type=nengo.neurons.LIF(tau_rc)
    )
    
    nengo.Connection(input_node, a)
    output_node = nengo.Node(size_in=dimensions
    )
    
    nengo.Connection(a, output_node, synapse=t_pstc, function=target_function)
    
import nengo_gui
nengo_gui.GUI(__file__).start()



