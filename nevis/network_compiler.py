import nengo
import math
import logging

from nengo.ensemble import Ensemble
from numpy.lib.utils import source
from nevis.neuron_classes import Synapses, Encoder
from nevis.config_tools import ConfigTools
from nevis.filetools import Filetools
import numpy as np
import nengo
from matplotlib import pyplot as plt

from nevis.neuron_classes import UART
from nevis.neuron_classes import InputNode, OutputNode, DirectConnection

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

        # Purge previous build caches 
        ConfigTools.purge_model_config()
        Filetools.purge_directory("nevis/file_cache")

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
        adj_mat_obj = adj_mat_params = adj_mat_visual  = np.zeros([node_num, node_num])

        # Declared seperately as numpy lists cannot be heterogenous.
        adj_mat_nevis = [[0.0 for _ in range(node_num)] for _ in range(node_num)]

        adj_mat_obj = adj_mat_obj.astype(nengo.Connection)

        param_class = type(param_model.params[model.connections[1]])
        adj_mat_params = adj_mat_params.astype(param_class)

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
        # 1. Begin iterating through the objects.
        # 2. Generate the objects.
        # 3. If ensemble, add it's Verilog wires to the top file.
        # 4. Generate the connections associated with the outputs 
        #   from the object.

        # Compile the parameters first, then compile the modules.
        # This will make absolutely sure that Verilog does not infer 
        # signals if this is not desired.

        uart_obj = UART(
            baud          = 2000000,
            n_input_data  = self.comp_args["bits_input"],
            n_output_data = self.comp_args["n_connection_output"]
        )

        nevis_top = open("nevis/sv_source/nevis_top.sv").read()

        obj_lst_nevis = []
        
        for i, vertex in enumerate(obj_lst_obj):
                
            if type(vertex) == nengo.Node:
            
                conns = adj_mat_obj[i][np.nonzero(adj_mat_visual[i])]

                if len(conns) != 0:
                    # The node is a source node - wait until after
                    # ensemble compilation to compile the data transfer
                    # hardware.
                    source_node = InputNode(
                        dims = vertex.size_out
                    )
                    uart_obj.in_node_dimensionalites.append(vertex.size_out)

                    obj_lst_nevis.append(source_node)

                    # TODO Combine this with the code below into a function.
                    conn_indices = np.nonzero(adj_mat_visual[i])[0]
                    conns = adj_mat_obj[i][conn_indices]

                    for node_data in zip(conns,conn_indices):

                        source_conn = DirectConnection(
                            dims = node_data[0].pre_obj.size_out
                        )

                        adj_mat_nevis[i][node_data[1]] = source_conn

                else:
                    sink_node = OutputNode(
                        dims = vertex.size_out,
                        index = i
                    )
                
                    pre_obj = np.nonzero(adj_mat_visual[:,i])[0]

                    if len(pre_obj) != 1:
                        print("ERROR: output nodes must only connect from one ensemble to one output node.")

                    sink_node.pre_objs.append(pre_obj[0])
                    uart_obj.out_nodes.append(sink_node)

                    obj_lst_nevis.append(sink_node)

            elif type(vertex) == nengo.Ensemble:
                
                # Count the number of inputs to the ensemble
                input_num = np.count_nonzero(adj_mat_obj[i])

                # The way that the Verilog interprets the connection
                # parameter files needs to be changed to:
                # weights_<preobj>C<postobj>_<dimension>

                # Generate the ensemble and add it's parameter 
                # declarations to the nevis_top file.
                ensemble = self.generate_nevis_ensemble(
                    ens_obj     = vertex, 
                    ens_params  = obj_lst_params[i], 
                    index       = i, 
                    input_num   = input_num
                )
                nevis_top += ensemble.verilog_wire_declaration()

                obj_lst_nevis.append(ensemble)
                
                conn_indices = np.nonzero(adj_mat_visual[i])[0]
                output_conns = adj_mat_obj[i][conn_indices]
                output_conn_params = adj_mat_params[i][conn_indices]
                
                for conn_data in zip(output_conns, output_conn_params, conn_indices):
                    connection = self.generate_nevis_connection(
                        conn_obj    = conn_data[0],
                        conn_params = conn_data[1],
                        pre_index   = i,
                        post_index  = conn_data[2]
                    )
                    nevis_top += connection.verilog_wire_declaration()
                    adj_mat_nevis[i][int(conn_data[2])] = connection

            else:
                print("ERROR")
                logger.error("[NeVIS]: Only node and ensemble objects are supported.")

        # CREATE THE UART OBJECT AND INSTANTIATE IT IN THE VERILOG.
        uart_obj.verilog_create_uart()

        # CREATE THE ENSEMBLE OBJECTS AND COMPILE THEIR CONNECTIONS.

        for i, ens in enumerate(obj_lst_nevis):
            if type(ens) == Encoder:
                
                # Declare the connections.
                fan_in_indices = np.nonzero(adj_mat_visual[i])[0]

                # Declare the ensemble in the Verilog.
                nevis_top += ens.verilog_mod_declaration()

                nevis_top += ens.verilog_input_declaration(
                    post_indices = fan_in_indices
                )

        exit()

        print(obj_lst_nevis)
        print(adj_mat_nevis)
        print(adj_mat_visual)
        # End the SystemVerilog module.
        nevis_top += "endmodule"
        print(nevis_top)

        #plt.matshow(adj_mat_visual)
        #plt.show()
        exit()

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

        pre_obj = conn_obj.pre_obj
        if type(pre_obj) == nengo.Ensemble:
            conn_args["pre_radius"] = pre_obj.radius
        else:
            conn_args["pre_radius"] = 1

        # Compile an output node (Nevis - Synapses)
        output_hardware = Synapses(
            pstc_scale      = conn_args["pstc_scale"],
            decoders_list   = conn_args["weights"], 
            radius_pre      = conn_args["pre_radius"],
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
            conn_obj    = network.connections[1],
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

