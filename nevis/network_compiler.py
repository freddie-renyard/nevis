import nengo
import math
import logging
from nevis.neuron_classes import Synapses_Floating, Encoder_Floating
from nevis.config_tools import ConfigTools
import numpy as np

logger = logging.getLogger(__name__)

class NetworkCompiler:

    @classmethod
    def compile_ensemble(cls, model, network):
        """ This function has many arguments prespecified, as full network compilation
        is not supported yet. These will gradually be phased out as more functionality
        is added.
        """

        # Gather simulation parameters - identical across all ensembles
        sim_args = {}
        sim_args["dt"] = model.dt

        # Define the compiler params. TODO write an optimiser function to
        # define these params automatically.
        comp_args = {}
        comp_args["radix_encoder"] = 4
        comp_args["bits_input"] = 8
        comp_args["radix_input"] = comp_args["bits_input"] - 1
        comp_args["radix_weights"] = 7
        comp_args["n_dv_post"] = 10
        comp_args["n_activ_extra"] = 6
        comp_args["min_float_val"] = 1*2**-50

        # Gather ensemble parameters - vary between ensembles
        ens_args = {}
        ens_args["n_neurons"] = network.ensemble.n_neurons
        ens_args["input_dimensions"] = network.input_dimensions
        ens_args["output_dimensions"] = network.output_dimensions
        ens_args["bias"] = model.params[network.ensemble].bias
        ens_args["t_rc"] = network.ensemble.neuron_type.tau_rc
        ens_args["t_rc"] = ens_args["t_rc"] / sim_args["dt"]
        # scaled_encoders = gain * encoders
        # TODO this is computationally wasteful, but the way that the Encoder 
        # object is designed at present makes the code below the most readable 
        # solution. Change the Encoder so that this is not the case.
        ens_args["encoders"] = model.params[network.ensemble].encoders
        ens_args["gain"] = model.params[network.ensemble].gain
        # Gather refractory period
        ens_args["ref_period"] = network.ensemble.neuron_type.tau_ref / sim_args["dt"]

        print(model.params[network.ensemble].gain)
        print(model.params[network.ensemble].encoders)
        print(model.params[network.ensemble].scaled_encoders)

        # Compile an ensemble (NeVIS - Encoder)
        input_hardware = Encoder_Floating(
            n_neurons=ens_args["n_neurons"],
            gain_list=ens_args["gain"],
            encoder_list=ens_args["encoders"],
            bias_list=ens_args["bias"],
            t_rc=ens_args["t_rc"],
            ref_period=ens_args["ref_period"],
            n_x=comp_args["bits_input"],
            radix_x=comp_args["radix_input"],
            radix_g=comp_args["radix_encoder"],
            radix_b=comp_args["radix_encoder"],
            n_dv_post=comp_args["n_dv_post"],
            index=0,
            verbose=False
        )

        # Tool for painlessly investigating the parameters of Nengo objects
        #l = dir(network.connections[1])
        
        conn_args = {}
        conn_args["weights"] = model.params[network.connection].weights
        conn_args["t_pstc"] = network.connections[0].synapse.tau
        conn_args["t_pstc"] = conn_args["t_pstc"] / sim_args["dt"]
        conn_args["pstc_scale"] = 1.0 - math.exp(-1.0 / conn_args["t_pstc"])
        conn_args["n_output"] = 10
        logger.info("t_pstc: %f", conn_args["t_pstc"])

        # Compile an output node (Nevis - Synapses)
        output_hardware = Synapses_Floating(
            n_neurons=ens_args["n_neurons"],
            pstc_scale=conn_args["pstc_scale"],
            decoders_list=conn_args["weights"], 
            encoders_list=[1], # Indicates a positive weight addition
            n_activ_extra=comp_args["n_activ_extra"],
            n_output=conn_args["n_output"],
            radix_w=comp_args["radix_weights"],
            minimum_val=comp_args["min_float_val"],
            pre_index=0,
            post_start_index=1,
            verbose=True
        )

        # Save the compiled models's parameters in a JSON file
        # TODO adapt this for higher dimensional representation.
        ConfigTools.create_model_config_file(
            in_node_depths= [input_hardware.n_x],
            out_node_depths= [output_hardware.n_output],
            out_node_scales= [output_hardware.n_output-4],
            n_values=np.shape(conn_args["weights"])[0]
        )

