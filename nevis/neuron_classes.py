from numpy.core.fromnumeric import nonzero
from nevis.memory_compiler import Compiler
import math
import numpy as np
import logging

from nevis.filetools import Filetools
from nevis.config_tools import ConfigTools
"""
This file contains the classes for the compiled sections of different parts of the network.
TODO Convert the fixed/floating distiction into subclasses of a main Encoder class etc.

Current setup:
1. Encoder (gain/bias --> NAU) (scalar -> spikes)
2. Synapses (SFU/WAC) (spikes -> scalar)

Proposed, more general setup:
1. Encoder (gain/bias --> NAU) (vector -> spikes)
2. Ensemble (SFU/WAC --> gain/bias --> NAU) (spikes -> spikes)
3. Decoder (SFU/WAC) (spikes -> vector)

These classes will have more attributes such as dimentionality of the
input, neuron type etc. which can be used to drop in modules into the 
full_model.sv file. The lower level modules will need more parameterisation
with if statements e.g. LIF neuron=0x0, instantiate LIF NAU. Determine 
whether the other neuron models e.g. Izhikevich neurons can be implemented
with a change solely to the NAU. Otherwise, full drop in high level modules
can be created specifically for each neuron.
"""

logger = logging.getLogger(__name__)

class Encoder:
    
    def __init__(self, 
            input_num, 
            gain_list, encoder_list, bias_list, 
            t_rc, ref_period, 
            n_x, radix_x, radix_g, radix_b, radix_phi, 
            index, 
            n_dv_post, 
            verbose):
        """ Creates the appropriate parameters needed for the encoder module in hardware. On initialisation, the 
        class runs the compilation of all the relevant model parameters and stores them
        as attributes of the instance of the class.

        Parameters
        ----------
        gain_list: [float]
            The gain parameters of the Nengo model.
        encoder_list: [Any]
            The encoder parameters of the Nengo model.
        bias_list: [float]
            The bias parameters of the Nengo model.
        t_rc: Any
            The RC constant of the neurons in the module. Despite the division 
            usually occuring after gain and bias mulitplication, the gains and 
            biases are divided before compilation to prevent the need for 
            division operations in hardware.
        ref_period : float or int
            The refractory period, normalised by the timestep.
        n_x: int
            The bit depth of the encoder's input value.
        radix_x: int
            The bit position of the radix point in x's representation.
        radix_g: int
            The place of the desired binary point of the gain's value. Used to 
            specify compiled fixed-point integer parameter precision.
        radix_b: int
            The place of the desired binary point of the bias's value. Used to 
            specify compiled fixed-point integer parameter precision.
        n_dv_post: int 
            The bit depth of the dV value after multiplication by the gain and bias.
            Used to control the precision of the activation in the NAU.
        verbose: False
            A flag that is passed to the model parameter compiler to check each
            gain/bias value against its fixed point couunterpart.
        
        Returns
        -------
        None.
        """

        self.radix_g_mantissa = radix_g
        self.radix_b_mantissa = radix_b

        # Range of x is confined to 0.99... to -1 (Q0.x)
        self.n_x = n_x
        self.radix_x = radix_x

        # Number of neurons that the ensemble represents.
        self.n_neurons = np.shape(encoder_list)[0]

        # Input vector dimensions.
        self.input_dims = np.shape(encoder_list)[-1]
        
        # Number of inputs (Fan-in)
        self.input_num = input_num

        # The index of this node in the context of the whole graph
        self.index = index

        if self.input_dims == 1:
            # Multiply the gains by their respective encoder values and divide by RC constant.
            self.eg_trc_list = [(x * y) / t_rc for x, y in zip(gain_list, encoder_list)]
            self.n_phi = 0
        else:
            # Omit the encoders from the multiplication, as the dimensionality is higher than 1.
            self.eg_trc_list = [x / t_rc for x in gain_list]
  
            self.encoders = encoder_list
            # Flatten the maxtrix to force consistent param compilation across all encoders.
            flat_encoders = self.encoders.flatten()
            
            logger.info(self.encoders)
            logger.info(flat_encoders)
            logger.info("Compiling phis...")
            self.compile_and_save_encoders(target_list = flat_encoders, radix_phi=radix_phi)

        self.b_trc_list = [x / t_rc for x in bias_list]

        # Compute params needed to represent the refractory period.
        self.ref_value, self.n_r = self.calculate_refractory_params(ref_period)

        # Compute the left shift needed to represent the neuronal RC constant.
        self.t_rc_shift = int(math.log2(t_rc))

        # Declaring class attributes to compile
        self.n_dv_post = n_dv_post

        # Compile the gain and bias into seperate lists.
        self.comp_gain_list, self.n_g_mantissa, self.n_g_exponent = Compiler.compile_to_float(self.eg_trc_list, self.radix_g_mantissa, verbose=verbose)
        self.comp_bias_list, self.n_b_mantissa, self.n_b_exponent = Compiler.compile_to_float(self.b_trc_list, self.radix_b_mantissa, verbose=verbose)

        self.save_params(index, floating=True)

    def compile_nau_start_params(self, n_voltage, n_neuron, n_ref): 
        """ Compiles the start file for the NAU.
        These will start at zero by default, and are also concatenated
        with the refractory period for each neuron.

        Parameters
        ----------
        n_voltage: int
            The bit depth of the activation datapath.
        n_neuron: int
            The number of neurons in the module.
        n_ref: int
            The bit depth of the refractory period, which are 
            concatenated onto the memory entry as the LSBs.

        Returns
        -------
        target_list: [str]
            The start parameters as a list of strings, prepared for output
            into a memory file.
        
        """
        target_list = []

        for i in range(n_neuron):
            target_list.append((n_voltage) * "0")

        for i in range(n_neuron):
            target_list[i] = target_list[i] + (n_ref * "1")

        return target_list
    
    def calculate_refractory_params(self, refractory):
        """Calculates the appropriate hardware parameters for the refractory period specified.

        Parameters
        ----------
        refractory : float
            The refractory period, normalised by the model's timestep.

        Returns
        -------
        period: int
            The new refractory period in context of the hardware implementation of the LIF spiking behaviour.
        bit_width: int
            The hardware bitwidth needed to store the value and produce the appropriate overflow
            for hardware refractory period behaviour.
        """
        
        bit_width = math.ceil(math.log2(refractory+1))
        refractory = 2 ** bit_width - refractory - 1

        return int(refractory), int(bit_width)
    
    def compile_and_save_encoders(self, target_list, radix_phi):
        """This method will determine if the encoder dimensionality is more
        than 1, and whether an encoder list will need to be compiled and saved.
        """
        self.comp_encoder_concat = [""] * (self.n_neurons)
        if self.input_dims != 1:
            self.comp_enc_list, self.n_phi = Compiler.compile_floats(target_list, radix_phi, verbose=True)
            comp_enc_struct = np.reshape(self.comp_enc_list, (self.input_dims, self.n_neurons))
            for compiled_str in comp_enc_struct:
                for i, string in enumerate(compiled_str):
                    self.comp_encoder_concat[i] = string + self.comp_encoder_concat[i]

    def save_params(self, index, floating=True, running_mem_total=0):

        index = str(index)

        # Save the encoder params
        filename = "encoder_compiled" + index + ".mem"
        logger.info("INFO: Saving gain and bias to binary .mem file as %s", filename)
        combined = Filetools.combine_binary_params(self.comp_gain_list, self.comp_bias_list)
        running_mem_total = Filetools.save_to_file(
            filename, 
            combined,
            running_mem_total
        )

        # Save the NAU start params
        filename = "nau_compiled" + index + ".mem"
        logger.info("INFO: Saving NAU start parameters to binary .mem file as %s", filename)
        combined = self.compile_nau_start_params(n_voltage=self.n_dv_post, n_ref=self.n_r, n_neuron=self.n_neurons)
        running_mem_total = Filetools.save_to_file(
            filename, 
            combined,
            running_mem_total
        )

        if self.input_dims != 1:
            filename = "phis_compiled" + index + ".mem"
            logger.info("INFO: Saving encoder phi vectors to binary .mem file as %s", filename)
            running_mem_total = Filetools.save_to_file(
                filename=filename,
                target_list=self.comp_enc_list,
                running_mem_total=running_mem_total
            )

        # Write all relevant params for this portion of the network to the Verilog header file.
        verilog_header = open("nevis/file_cache/model_params.vh", "a")

        verilog_header.write(("// Population " + index + ' Params' + '\n'))
        verilog_header.write(('parameter ' + 'N_NEURON_' + index + ' = ' + str(self.n_neurons) + ',' + '\n'))
        # X/Incoming activation Params
        verilog_header.write(('N_X_' + index + ' = ' + str(self.n_x) + ',' + '\n'))
        verilog_header.write(('RADIX_X_' + index + ' = ' + str(self.radix_x) + ',' + '\n'))

        # Dot product params
        verilog_header.write(('N_PHI_' + index + ' = ' + str(self.n_phi) + ',' + '\n'))
        verilog_header.write(('INPUT_DIMS_' + index + ' = ' + str(self.input_dims) + ',' + '\n'))
        verilog_header.write(('INPUT_NUM_' + index + ' = ' + str(self.input_num) + ',' + '\n'))

        # NAU Params
        verilog_header.write(('N_R_' + index + ' = ' + str(self.n_r)  + ',' + '\n'))
        verilog_header.write(('REF_VALUE_' + index + ' = ' + str(self.ref_value)  + ',' + '\n'))
        verilog_header.write(('T_RC_SHIFT_' + index + ' = ' + str(self.t_rc_shift)  + ',' + '\n'))
        # Output value to NAU param
        verilog_header.write(('N_DV_POST_' + index + ' = ' + str(self.n_dv_post) + ',' + '\n'))

        if floating:
             # Gain Params
            verilog_header.write(('N_G_MAN_' + index + ' = ' + str(self.n_g_mantissa) + ',' + '\n'))
            verilog_header.write(('N_G_EXP_' + index + ' = ' + str(self.n_g_exponent) + ',' + '\n'))
            # Bias Params
            verilog_header.write(('N_B_MAN_' + index + ' = ' + str(self.n_b_mantissa) + ',' + '\n'))
            verilog_header.write(('N_B_EXP_' + index + ' = ' + str(self.n_b_exponent) + ';' + '\n'))
        else:
            # Gain Params
            verilog_header.write(('N_G_' + index + ' = ' + str(self.n_g) + ',' + '\n'))
            verilog_header.write(('RADIX_G_' + index + ' = ' + str(self.radix_g) + ',' + '\n'))
            # Bias Params
            verilog_header.write(('N_B_' + index + ' = ' + str(self.n_b) + ',' + '\n'))
            verilog_header.write(('RADIX_B_' + index + ' = ' + str(self.radix_b) + ';' + '\n'))

        verilog_header.write('\n')
        verilog_header.close()

        return running_mem_total

    def verilog_wire_declaration(self):    
        
        output_str = open("nevis/sv_source/ensemble_wires.sv").read()
        return output_str.replace("<i>", str(self.index))

    def verilog_mod_declaration(self):

        output_str = open("nevis/sv_source/ensemble_mod.sv").read()
        return output_str.replace("<i>", str(self.index))

    def verilog_input_declaration(self, post_indices):

        assignment = open("nevis/sv_source/ensemble_ins.sv").read()
        output_str = ""
        for i, index in enumerate(post_indices):
            output_str += assignment.replace("<current_i>", str(i)).replace("<i_pre>", str(index)).replace("<i_post>", str(self.index))
        output_str += "\n"
        return output_str
        
class Synapses:

    def __init__(self, 
        pstc_scale, 
        decoders_list, radius_pre,
        n_activ_extra, n_output, radix_w, 
        minimum_val, 
        pre_index, post_start_index, 
        verbose=False):
        """ Creates the appropriate parameters needed for the synaptic weights module in hardware. 
        On initialisation, the class runs the compilation of all the relevant model parameters and 
        stores them as attributes of the instance of the class.

        Parameters
        ----------
        pstc_scale: float
            The model's scaling factor for the post-synaptic filter. For optimal 
            results, the reciprocal of this value must be a power of two as the hardware
            is built to divide by using right shifting. If the value is != 1/(2^n), the 
            method will round to the nearest 2^n, producing suboptimal results.
        decoders_list: Ndarray, shape=(dimensions, decoders)
            The decoder parameters of the Nengo model.
        radius_pre: int or float
            The radius of the ensemble which feeds spikes to this decoder. This allows
            for prevention of overflow. 
        encoders_list: [Any]
            The encoder parameters of the Nengo model.
        n_activ_extra: int
            The number of extra bits with which to compile the activation. This 
            allows for the post synaptic filter to have more precision when performed.
        radix_w: int
            The place of the desired binary point of the weights' value. Used to 
            specify compiled fixed-point integer parameter precision.
        verbose: False
            A flag that is passed to the model parameter compiler to check each
            weight value against its fixed point couunterpart.
        
        Returns
        -------
        None.
        """

        decoders_list = decoders_list * 1
        self.output_dims = np.shape(decoders_list)[0]

        self.n_neurons_pre = np.shape(decoders_list)[1]

        self.pre_index = pre_index
        self.post_index = post_start_index
        
        # Clip small values to reduce dynamic range and hence decrease required exponent bit depth.
        if minimum_val != 0:
            for i, weight_list in enumerate(decoders_list):
                decoders_list[i] = [Compiler.clip_value(x, minimum_val) for x in weight_list]

        self.scale_w = Compiler.determine_middle_exp(decoders_list.flatten())

        extra_bits = math.ceil(math.log2(radius_pre)) - 1
        if extra_bits < 0: extra_bits = 0
        self.scale_w += extra_bits
        
        self.radix_w = radix_w
        self.n_output = n_output
        self.n_activ_extra = n_activ_extra
        
        # Calculate the number of bits to shift by to implement the post_synaptic filter.
        n_value = 1 / pstc_scale
        self.pstc_shift = int(math.log2(n_value))

        # Multiply weights by scale factor
        scale_factor_w = 2 ** self.scale_w
        logger.info(decoders_list)
        self.weights = decoders_list * scale_factor_w
        logger.info(self.weights)

         # Compile the weights
        highest_exp = 0

        # Calculate the largest exponent needed for each weight list, to ensure the same bit depths across weights.
        for weight_list in self.weights:
            new_exp = Compiler.calculate_exponent_depth(weight_list)
            if new_exp > highest_exp:
                highest_exp = new_exp

        self.comp_weights_list = []
        self.n_w_man = None
        self.n_w_exp = None
        for i, weight_list in enumerate(self.weights):
            comp_weights, n_w_man, n_w_exp = Compiler.compile_to_float(weight_list, self.radix_w, highest_exp, verbose=verbose)
            logger.info("Weights {}".format(i+1))
            logger.info("{} {}".format(n_w_man, n_w_exp))
            self.comp_weights_list.append(comp_weights)

            if self.n_w_man is not None or self.n_w_exp is not None:
                if n_w_man != self.n_w_man or n_w_exp != self.n_w_exp:
                    logger.error("The compiled bit parameters do not match between weight lists. This will cause unpredictable hardware behaviour.")
                    logger.info("Recompiling weights with hard limit on higher exponent...")
            
            self.n_w = n_w_exp + n_w_man
            self.n_w_man, self.n_w_exp = n_w_man, n_w_exp

        self.save_params(pre_index, post_start_index, floating=True)

    def save_params(self, pre_index, post_start_index, floating=True, running_mem_total=0):
        
        index = str(pre_index) + "C"

        # Save each decoder's compiled weight list to a .mem file.
        for i, compiled_lists in enumerate(self.comp_weights_list):

            filename = "weights_compiled" + index + str(post_start_index+i) + ".mem"
            running_mem_total = Filetools.save_to_file(
                filename, 
                compiled_lists,
                running_mem_total
            )

        index += str(post_start_index)

        # Write all relevant params for this portion of the network to the Verilog header file.
        verilog_header = open("nevis/file_cache/model_params.vh", "a")

        verilog_header.write(("// Decoder " + index + ' Params' + '\n'))
        verilog_header.write(('parameter ' + 'N_WEIGHT_' + index + ' = ' + str(self.n_w) + ',' + '\n'))
        verilog_header.write(('SCALE_W_' + index + ' = ' + str(self.scale_w) + ',' + '\n'))
        verilog_header.write(('N_NEURON_PRE_' + index + ' = ' + str(self.n_neurons_pre) + ',' + '\n'))
        verilog_header.write(('N_ACTIV_EXTRA_' + index + ' = ' + str(self.n_activ_extra) + ',' + '\n'))
        verilog_header.write(('N_OUTPUT_' + index + ' = ' + str(self.n_output) + ',' + '\n'))
        verilog_header.write(('PSTC_SHIFT_' + index + ' = ' + str(self.pstc_shift) + ',' + '\n'))
        verilog_header.write(('OUTPUT_DIMS_' + index + ' = ' + str(self.output_dims) + ',' + '\n'))

        if floating:
            verilog_header.write(('N_WEIGHT_EXP_' + index + ' = ' + str(self.n_w_exp) + ';' + '\n'))
        else:
            verilog_header.write(('N_WEIGHT_EXP_' + index + ' = ' + str(0) + ';' + '\n'))

        verilog_header.write('\n')
        verilog_header.close()

        return running_mem_total
    
    def verilog_wire_declaration(self):    
        
        output_str = open("nevis/sv_source/connection_wires.sv").read()
        output_str = output_str.replace("<i_pre>", str(self.pre_index))
        return output_str.replace("<i_post>", str(self.post_index))

    def calculate_scale_w(self, target_list):

        max_val = np.amax([abs(x) for x in target_list])
        scale_val = 0

        while max_val <= 1.0:
            scale_val += 1
            max_val *= 2.0
            print(max_val)

        return scale_val-1

class InputNode:

    def __init__(self, dims, index, post_objs):
        self.dims = dims
        self.index = index

        self.post_objs = post_objs
        
class OutputNode:

    def __init__(self, dims, index):
        self.dims = dims
        self.index = index

        self.pre_objs = []

class DirectConnection:

    def __init__(self, dims, pre_index, post_index):

        self.dims = dims
        self.pre_index = pre_index
        self.post_index = post_index

        self.save_params()

    # TODO Give connections class inheritance to prevent code duplication.
    def save_params(self, running_mem_total=0):
        
        index = str(self.pre_index) + "C"

        index += str(self.post_index)

        # Write all relevant params for this portion of the network to the Verilog header file.
        verilog_header = open("nevis/file_cache/model_params.vh", "a")

        verilog_header.write(("// Connection " + index + ' Params' + '\n'))
        verilog_header.write(('parameter ' + 'N_OUTPUT_' + index + ' = ' + str(1) + ',' + '\n'))
        verilog_header.write(('OUTPUT_DIMS_' + index + ' = ' + str(self.dims) + ';' + '\n'))

        verilog_header.write('\n')
        verilog_header.close()

        return running_mem_total
    
    def verilog_wire_declaration(self):    
        
        output_str = open("nevis/sv_source/connection_wires.sv").read()
        output_str = output_str.replace("<i_pre>", str(self.pre_index))
        return output_str.replace("<i_post>", str(self.post_index))
        
class UART:

    def __init__(self, baud, n_input_data, n_output_data):

        # The width of the input (tx) and output (rx) data words.
        # This is before they are flattened into a bitstream and 
        # sent to the FPGA. These must be consistent across all
        # I/O nodes.
        self.n_input_data = int(n_input_data)
        self.n_output_data = int(n_output_data)
        self.baud = baud

        self.in_node_dimensionalites = []
        
        self.out_nodes = []
        self.in_nodes = []

        self.save_params()

    def save_params(self):

        with open("nevis/file_cache/model_params.vh", "a") as verilog_header:
            verilog_header.write("// UART parameters")
            verilog_header.write("parameter N_RX = " + str(self.n_input_data) + ",\n")
            verilog_header.write("N_TX = " + str(self.n_output_data) + ",\n")
            verilog_header.write("TX_NUM_OUTS = " + str(len(self.out_nodes)) + ",\n")
            verilog_header.write("RX_NUM_INS = " + str(len(self.in_nodes)) + ",\n")
            verilog_header.write("BAUD_RATE = " + str(self.baud) + ";\n\n")

    def verilog_create_uart(self):

        verilog_out = open('nevis/sv_source/uart_wires_mod.sv').read()

        # Add the output (TX) assignments
        tx_assignment = open('nevis/sv_source/uart_tx_ins.sv').read()
        
        bit_pointer = 0
        for out_node in self.out_nodes:
            for obj in out_node.pre_objs:
                for dim in range(out_node.dims):
                    assign = tx_assignment.replace("<i_pre>", str(obj))
                    assign = assign.replace("<i_post>", str(out_node.index))
                    assign = assign.replace("<i_dim>", str(dim))

                    assign = assign.replace("<bit_pre>", str(bit_pointer))
                    assign = assign.replace("<bit_post>", str(bit_pointer + self.n_output_data))

                    verilog_out = verilog_out.replace("<tx-flag>", assign)

                    bit_pointer += self.n_output_data
        
        output_num = bit_pointer % self.n_output_data

        verilog_out = verilog_out.replace("<tx-flag>", "\n")

        # Add the input (RX) assigments
        rx_assignment = open('nevis/sv_source/uart_rx_ins.sv').read()
        
        # TODO The code below will not work with part-selected inputs as 
        # the code assumes consistent dimensionality between input connections.
        # Add this functionality by passing connection dimensionality data to
        # the input node.
        bit_pointer = 0
        for in_node in self.in_nodes:
            for post_index in in_node.post_objs:
                # The following assumes constant connection dimensionality
                for dim in range(in_node.dims):
                    assign = rx_assignment.replace("<i_pre>", str(in_node.index))
                    assign = assign.replace("<i_post>", str(post_index))
                    assign = assign.replace("<i_dim>", str(dim))

                    assign = assign.replace("<bit_pre>", str(bit_pointer))
                    assign = assign.replace("<bit_post>", str(bit_pointer + self.n_output_data))

                    verilog_out = verilog_out.replace("<rx-flag>", assign)

                    bit_pointer += self.n_output_data

        input_num = bit_pointer % self.n_input_data

        verilog_out = verilog_out.replace("<rx-flag>", "")

        ConfigTools.create_model_config_file(
            in_node_depth = self.n_input_data,
            out_node_depth=self.n_output_data,
            out_node_scale=self.n_output_data - 4,
            n_input_values=input_num,
            n_output_values=output_num
        )

        return verilog_out

