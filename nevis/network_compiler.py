import nengo
import math
import logging
from nevis.neuron_classes import Synapses_Floating, Encoder_Floating
from nevis.config_tools import ConfigTools
import numpy as np
import nengo
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

class NevisCompiler:

    @classmethod
    def compile_network(cls, model):
        """Builds a model into a NeVIS network representation, ready for synthesis.
        """

        # Build the model, to allow for parameter exrtraction.
        param_model = nengo.builder.Model()
        nengo.builder.network.build_network(param_model, model)
        
        print()
        
        # Create two object lists which hold the objects and
        # their associated built parameters.
        obj_lst_obj = []
        obj_lst_params = []

        # Add all nodes on the network graph to a list.
        for node in model.nodes:
            obj_lst_obj.append(node)
            obj_lst_params.append(param_model.params[node])

        for ensemble in model.ensembles:
            obj_lst_obj.append(ensemble)
            obj_lst_params.append(param_model.params[ensemble])

        # Create two adjacency matrices which correspond to
        # the list created above. These contain the Connections
        # and BuiltConnections.
        node_num = len(obj_lst_obj)
        adj_mat_obj = adj_mat_params = adj_mat_visual = np.zeros([node_num, node_num])
        
        adj_mat_obj = adj_mat_obj.astype(nengo.Connection)

        param_class = type(param_model.params[model.connections[1]])
        adj_mat_params = adj_mat_params.astype(param_class)
        
        #print(model.connections)
        #print(param_model.params[model.ensembles[0]])

        # A note on convention: the first index of the 
        # adjacency matrix refers to the 'source' object 
        # on the directed graph, and the second index 
        # refers to the 'sink' e.g. a connection at 
        # adj_mat[1][2] would indicate a connection
        # from node at index 1 in obj_lst to the node
        # at index 2.
        for edge in model.connections:
            for i, pre_node in enumerate(obj_lst_obj):
                if edge.pre_obj == pre_node:
                    for j, post_node in enumerate(obj_lst_obj):
                        if edge.post_obj == post_node:
                            adj_mat_obj[i][j] = edge
                            adj_mat_params[i][j] = param_model.params[edge]
                            adj_mat_visual[i][j] = 1
        
        # High level overview:
        # 1. Begin iterating through the source objects.
        # 2. Generate the source object.
        # 3. Add its Verilog template to the open nevis_top.sv file
        # 4. Iterate over its connections.
        # 5. Generate the connection object.
        # 6. Add its Verilog template to the open nevis_top.sv file
        # A note on compiling Verilog: have the scripts inside the 
        #   objects, but pass in some connection based parameters from 
        #   outside.

        #plt.matshow(adj_mat_visual)
        #plt.show()

        print()

    @classmethod
    def generate_nevis_ensemble(cls, ens_obj, ens_params):
        """This method inputs a Nengo ensemble (both object and built obj if needed)
        and returns a NeVIS Encoder object.
        """
        print()

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
        comp_args["radix_encoder"]  = 10
        comp_args["bits_input"]     = 8
        comp_args["radix_input"]    = comp_args["bits_input"] - 1
        comp_args["radix_phi"]      = 5
        comp_args["radix_weights"]  = 7
        comp_args["n_dv_post"]      = 10
        comp_args["n_activ_extra"]  = 3
        comp_args["min_float_val"]  = 1*2**-50

        # Gather ensemble parameters - vary between ensembles
        ens_args = {}
        ens_args["n_neurons"]           = network.ensemble.n_neurons
        ens_args["input_dimensions"]    = network.input_dimensions
        ens_args["output_dimensions"]   = network.output_dimensions
        ens_args["bias"]                = model.params[network.ensemble].bias
        ens_args["t_rc"]                = network.ensemble.neuron_type.tau_rc
        ens_args["t_rc"]                = ens_args["t_rc"] / sim_args["dt"]
        
        # scaled_encoders = gain * encoders
        # TODO this is computationally wasteful, but the way that the Encoder 
        # object is designed at present makes the code below the most readable 
        # solution. Change the Encoder so that this is not the case.
        ens_args["encoders"]    = model.params[network.ensemble].encoders
        ens_args["gain"]        = model.params[network.ensemble].gain
        # Gather refractory period
        ens_args["ref_period"]  = network.ensemble.neuron_type.tau_ref / sim_args["dt"]

        # Compile an ensemble (NeVIS - Encoder)
        input_hardware = Encoder_Floating(
            n_neurons       = ens_args["n_neurons"],
            input_num       = 1, # cannot calculate this for single ensembles.
            gain_list       = ens_args["gain"],
            encoder_list    = ens_args["encoders"],
            bias_list       = ens_args["bias"],
            t_rc            = ens_args["t_rc"],
            ref_period      = ens_args["ref_period"],
            n_x             = comp_args["bits_input"],
            radix_x         = comp_args["radix_input"],
            radix_g         = comp_args["radix_encoder"],
            radix_b         = comp_args["radix_encoder"],
            radix_phi       = comp_args["radix_phi"],
            n_dv_post       = comp_args["n_dv_post"],
            index           = 0,
            verbose=False
        )

        # Tool for painlessly investigating the parameters of Nengo objects
        #l = dir(network.connections[1])
        
        conn_args = {}
        conn_args["weights"]    = model.params[network.connection].weights
        conn_args["t_pstc"]     = network.connections[0].synapse.tau
        conn_args["t_pstc"]     = conn_args["t_pstc"] / sim_args["dt"]
        conn_args["pstc_scale"] = 1.0 - math.exp(-1.0 / conn_args["t_pstc"])
        conn_args["n_output"]   = 10
        logger.info("t_pstc: %f", conn_args["t_pstc"])

        # Compile an output node (Nevis - Synapses)
        output_hardware = Synapses_Floating(
            n_neurons       = ens_args["n_neurons"],
            pstc_scale      = conn_args["pstc_scale"],
            decoders_list   = conn_args["weights"], 
            encoders_list   = [1], # Indicates a positive weight addition
            n_activ_extra   = comp_args["n_activ_extra"],
            n_output        = conn_args["n_output"],
            radix_w         = comp_args["radix_weights"],
            minimum_val     = comp_args["min_float_val"],
            pre_index       = 0,
            post_start_index= 1,
            verbose         = True
        )

        # Save the compiled models's parameters in a JSON file
        # TODO adapt this for higher dimensional representation.
        ConfigTools.create_model_config_file(
            in_node_depth   = input_hardware.n_x,
            out_node_depth  = output_hardware.n_output,
            out_node_scale  = output_hardware.n_output-4,
            n_input_values  = 1,
            n_output_values = np.shape(conn_args["weights"])[0]
        )

