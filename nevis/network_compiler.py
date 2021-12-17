import nengo
import math
import logging
from nevis.neuron_classes import Synapses, Encoder
from nevis.config_tools import ConfigTools
import numpy as np
import nengo
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

class NevisCompiler:

    def __init__(self):

        # Define the compiler params. TODO write some profiles which will allow
        # for a selection of different modes - e.g. high precision, efficient
        # with space, etc.
        self.comp_args = {}
        self.comp_args["radix_encoder"]  = 10
        self.comp_args["bits_input"]     = 8
        self.comp_args["radix_phi"]      = 5
        self.comp_args["radix_weights"]  = 7
        self.comp_args["n_dv_post"]      = 10
        self.comp_args["n_activ_extra"]  = 3
        self.comp_args["n_connection_output"]   = 10

        self.comp_args["min_float_val"]  = 1*2**-50

        # Gather simulation parameters - identical across all ensembles
        self.sim_args = {}
        self.sim_args["dt"] = 0.001 # Default value; overridden in main compiler.

    def compile_network(self, model):
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

    def generate_nevis_ensemble(self, ens_obj, ens_params, index, input_num):
        """This method inputs a Nengo ensemble (both object and built obj if needed)
        and returns a NeVIS Encoder object.
        """

        # Gather ensemble parameters - vary between ensembles
        ens_args = {}
        ens_args["input_dimensions"]    = ens_obj.size_in
        ens_args["output_dimensions"]   = ens_obj.size_out
        ens_args["bias"]                = ens_params.bias
        ens_args["t_rc"]                = ens_obj.neuron_type.tau_rc
        ens_args["t_rc"]                = ens_args["t_rc"] / self.sim_args["dt"]
        ens_args["radix_input"]         = self.comp_args["bits_input"] - 1 # TODO Needs to take into account the radius.
 
        # scaled_encoders = gain * encoders
        # TODO this is computationally wasteful, but the way that the Encoder 
        # object is designed at present makes the code below the most readable 
        # solution. Change the Encoder so that this is not the case.
        ens_args["encoders"]    = ens_params.encoders
        ens_args["gain"]        = ens_params.gain
        # Gather refractory period
        ens_args["ref_period"]  = ens_obj.neuron_type.tau_ref / self.sim_args["dt"]

        # Compile an ensemble (NeVIS - Encoder)
        nevis_ensemble = Encoder(
            input_num       = input_num, 
            gain_list       = ens_args["gain"],
            encoder_list    = ens_args["encoders"],
            bias_list       = ens_args["bias"],
            t_rc            = ens_args["t_rc"],
            ref_period      = ens_args["ref_period"],
            n_x             = self.comp_args["bits_input"],
            radix_x         = ens_args["radix_input"],
            radix_g         = self.comp_args["radix_encoder"],
            radix_b         = self.comp_args["radix_encoder"],
            radix_phi       = self.comp_args["radix_phi"],
            n_dv_post       = self.comp_args["n_dv_post"],
            index           = index,
            verbose=False
        )

        return nevis_ensemble

    def generate_nevis_connection(self, conn_obj, conn_params, pre_index, post_index):
        """This method generates a Connection (an edge on the 
        Nengo graph); Returns a NeVIS Connection object (Synapses).
        """

        conn_args = {}
        conn_args["weights"]    = conn_params.weights
        conn_args["t_pstc"]     = conn_obj.synapse.tau
        conn_args["t_pstc"]     = conn_args["t_pstc"] / self.sim_args["dt"]
        conn_args["pstc_scale"] = 1.0 - math.exp(-1.0 / conn_args["t_pstc"])

        # Compile an output node (Nevis - Synapses)
        output_hardware = Synapses(
            pstc_scale      = conn_args["pstc_scale"],
            decoders_list   = conn_args["weights"], 
            n_activ_extra   = self.comp_args["n_activ_extra"],
            n_output        = self.comp_args["n_connection_output"],
            radix_w         = self.comp_args["radix_weights"],
            minimum_val     = self.comp_args["min_float_val"],
            pre_index       = pre_index,
            post_start_index= post_index,
            verbose         = True
        )

        return output_hardware

    def compile_ensemble(self, model, network):
        """ This function has many arguments prespecified, as full network compilation
        is not supported yet. These will gradually be phased out as more functionality
        is added.
        """

        input_hardware = self.generate_nevis_ensemble(
            ens_obj     = network.ensemble,
            ens_params  = model.params[network.ensemble],
            index       = 0,
            input_num   = 1
        )

        # Tool for painlessly investigating the parameters of Nengo objects
        #l = dir(network.connections[1])
        
        output_hardware = self.generate_nevis_connection(
            conn_obj    = network.connections[0],
            conn_params = model.params[network.connection],
            pre_index   = 0,
            post_index  = 1
        )
        
        # Save the compiled models's parameters in a JSON file
        # TODO adapt this for higher dimensional representation.
        ConfigTools.create_model_config_file(
            in_node_depth   = input_hardware.n_x,
            out_node_depth  = output_hardware.n_output,
            out_node_scale  = output_hardware.n_output-4,
            n_input_values  = 1,
            n_output_values = output_hardware.output_dims
        )

