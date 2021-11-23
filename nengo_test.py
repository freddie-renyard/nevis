import nengo
from nengo.network import Network
from nevis.global_tools import Global_Tools
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np

# Define the input function to the neural population
def input_func(t):
    return np.sin(t *0.5* 2*np.pi), np.sin(t *0.5* 2*np.pi)

# Define, build and run a simple NeVIS model
def run_simple_fpga_model():
    with model:

        input_node = nengo.Node(input_func)

        t_pstc = Global_Tools.inverse_pstc(128, 0.001)
        tau_rc = Global_Tools.inverse_rc(8, 0.001)

        fpga_ens = NevisEnsembleNetwork(
            n_neurons=50,
            dimensions=1,
            compile_design=True,
            t_pstc=t_pstc,
            tau_rc=tau_rc
        )
        
        nengo.Connection(input_node, fpga_ens.input)
        #nengo.Connection(fpga_ens.output, fpga_ens.input)
        
        a = nengo.Ensemble(n_neurons=50, 
            dimensions=1, 
            neuron_type=nengo.neurons.LIF(tau_rc)
        )

        nengo.Connection(input_node, a)
        output_node = nengo.Node(size_in=1)
        nengo.Connection(a, output_node, synapse=t_pstc)

# Define, build and run a default Nengo network
def run_default_model():
    with model:
        stim = nengo.Node([0])
        a = nengo.Ensemble(n_neurons=50, 
            dimensions=1
        )
        nengo.Connection(stim, a)

model = nengo.Network()

run_simple_fpga_model()

import nengo_gui
nengo_gui.GUI(__file__).start()



