import nengo
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np



def input_func(t):
    return np.sin(t * 2*np.pi)

"""
with nengo.Network() as model:

    input_node = nengo.Node(input_func)

    fpga_ens = NevisEnsembleNetwork(
        n_neurons=50,
        dimensions=1,
        fpga_serial_addr="/dev/tty.usbserial-FT4ZS6I31",
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
"""

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim, a)
import nengo_gui
nengo_gui.GUI(__file__).start()



