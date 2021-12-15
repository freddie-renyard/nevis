import nengo
from nengo.network import Network
from numpy.core.numeric import True_
from nevis.global_tools import Global_Tools
from nevis.nevis_networks import NevisEnsembleNetwork
import numpy as np

# Define the input function to the neural population
def input_func(t):
    return 1 #np.sin(t * 2*np.pi)

def target_function(x):
    return x

model = nengo.Network()

with model:

    stim = nengo.Node(input_func)

    t_pstc = Global_Tools.inverse_pstc(128, 0.001)
    tau_rc = Global_Tools.inverse_rc(8, 0.001)

    dimensions = 1
    neuron_n = 10

    fpga_ens = NevisEnsembleNetwork(
        n_neurons=neuron_n,
        dimensions=dimensions,
        compile_design=False,
        t_pstc=t_pstc,
        tau_rc=tau_rc,
        function=target_function
    )
    nengo.Connection(stim, fpga_ens.input)
    #nengo.Connection(fpga_ens.output, fpga_ens.input)

    nevis_out = nengo.Probe(fpga_ens.input)

    a = nengo.Ensemble(
        n_neurons=neuron_n, 
        dimensions=dimensions,
        neuron_type=nengo.neurons.LIF(tau_rc)
    )

    spikes = nengo.Probe(a.neurons)

    nengo.Connection(stim, a)
    #nengo.Connection(a, a)
    print(a.probeable)
    output_node = nengo.Node(size_in=dimensions)
    in_probe = nengo.Probe(a, attr='input')
    out_probe = nengo.Probe(output_node)
    
    nengo.Connection(a, output_node, synapse=t_pstc, function=target_function)

with nengo.Simulator(model) as sim:
    for step in range(1000):
        sim.step()
        spike_lst = [i for i, x in enumerate(sim.data[spikes][-1] > 0) if x]
        print("Timestep:           ", step)
        print("Spikes:             ", spike_lst)

        print("Nengo input value: ", sim.data[in_probe][-1])
        print("Nengo output value: ", sim.data[out_probe][-1])

        print("NeVIS output value: ", sim.data[nevis_out][-1])
        input(" ")