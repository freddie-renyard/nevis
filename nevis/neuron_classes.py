from nevis.memory_compiler import Compiler
import math
import numpy as np
import logging

from nevis.filetools import Filetools
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
    
    def __init__(self, n_neurons, input_num, gain_list, encoder_list, bias_list, t_rc, ref_period, n_x, radix_x, radix_phi, n_dv_post):
        """ Creates the appropriate parameters needed for the encoder module in hardware. On initialisation, the 
        class runs the compilation of all the relevant model parameters and stores them
        as attributes of the instance of the class.

        Parameters
        ----------
        n_neurons: int
            The number of neurons in the module.
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

        # Range of x is confined to 0.99... to -1 (Q0.x)
        self.n_x = n_x
        self.radix_x = radix_x

        self.n_neurons = n_neurons

        # Input vecotr dimensions.
        self.input_dims = np.shape(encoder_list)[-1]
        
        # Number of inputs (Fan-in)
        self.input_num = input_num

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

    def compile_nau_start_params(self, n_activ, n_neuron, n_ref): 
        """ Compiles the start file for the NAU.
        These will start at zero by default, and are also concatenated
        with the refractory period for each neuron.

        Parameters
        ----------
        n_activ: int
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
            target_list.append((n_activ) * "0")

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
        combined = self.compile_nau_start_params(n_activ=self.n_dv_post, n_ref=self.n_r, n_neuron=self.n_neurons)
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

class Encoder_Fixed(Encoder):
    def __init__(self, n_neurons, input_num, gain_list, encoder_list, bias_list, t_rc, ref_period, n_x, radix_x, radix_g, radix_b, radix_phi, n_dv_post, index, verbose=False):
        self.radix_g = radix_g
        self.radix_b = radix_b
        super().__init__(self, n_neurons, input_num, gain_list, encoder_list, bias_list, t_rc, ref_period, n_x, radix_x, radix_phi, n_dv_post)

        # Compile the gain and bias into seperate lists.
        self.comp_gain_list, self.n_g = Compiler.compile_floats(self.eg_trc_list, self.radix_g, verbose=verbose)
        self.comp_bias_list, self.n_b = Compiler.compile_floats(self.b_trc_list, self.radix_b, verbose=verbose)

        self.save_params(index, floating=False)

class Encoder_Floating(Encoder):
    def __init__(self, n_neurons, input_num, gain_list, encoder_list, bias_list, t_rc, ref_period, n_x, radix_x, radix_g, radix_b, radix_phi, n_dv_post, index, verbose=False):

        self.radix_g_mantissa = radix_g
        self.radix_b_mantissa = radix_b

        super().__init__(n_neurons, input_num, gain_list, encoder_list, bias_list, t_rc, ref_period, n_x, radix_x, radix_phi, n_dv_post)

        # Compile the gain and bias into seperate lists.
        exp_limit = 15
        self.comp_gain_list, self.n_g_mantissa, self.n_g_exponent = Compiler.compile_to_float(self.eg_trc_list, self.radix_g_mantissa, exp_limit, verbose=verbose)
        self.comp_bias_list, self.n_b_mantissa, self.n_b_exponent = Compiler.compile_to_float(self.b_trc_list, self.radix_b_mantissa, exp_limit, verbose=verbose)

        
        self.save_params(index, floating=True)

class Synapses:

    def __init__(self, n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, n_output, radix_w, scale_w, verbose=False):
        """ Creates the appropriate parameters needed for the synaptic weights module in hardware. 
        On initialisation, the class runs the compilation of all the relevant model parameters and 
        stores them as attributes of the instance of the class.

        Parameters
        ----------
        n_neurons: int
            The number of neurons in the module.
        pstc_scale: float
            The model's scaling factor for the post-synaptic filter. For optimal 
            results, the reciprocal of this value must be a power of two as the hardware
            is built to divide by using right shifting. If the value is != 1/(2^n), the 
            method will round to the nearest 2^n, producing suboptimal results.
        decoders_list: Ndarray, shape=(dimensions, decoders)
            The decoder parameters of the Nengo model.
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
        
        self.radix_w = radix_w
        self.scale_w = scale_w
        self.n_output = n_output

        # This value is usually compiled in the Verilog, but is needed to compile
        # the starting memory files.
        self.n_activ_extra = n_activ_extra
        self.n_activ = 1 + self.radix_w + self.scale_w + n_activ_extra
        
        # Calculate the number of bits to shift by to implement the post_synaptic filter.
        n_value = 1 / pstc_scale
        self.pstc_shift = int(math.log2(n_value))
        
        self.n_neurons_pre = n_neurons

        # Multiply weights by scale factor
        scale_factor_w = 2 ** self.scale_w
        logger.info(decoders_list)
        self.weights = decoders_list * scale_factor_w
        logger.info(self.weights)
        
        self.comp_activation = []
        for _ in range(self.output_dims):
            self.comp_activation.append(self.compile_activations(n_activ=self.n_activ, n_neuron=1))
    
    def compile_activations(self, n_activ, n_neuron): 
        """ Compiles the start file for the synaptic activations.
        These will start at zero by default, and are also concatenated
        with the refractory period for each neuron.

        Parameters
        ----------
        n_activ: int
            The bit depth of the activation datapath.
        n_neuron: int
            The number of neurons in the module.

        Returns
        -------
        target_list: [str]
            The start parameters as a list of strings, prepared for output
            into a memory file.
        
        """
        target_list = []
        for i in range(n_neuron):
            target_list.append((n_activ) * "0")

        return target_list

    def save_params(self, pre_index, post_start_index, floating=True, running_mem_total=0):
        
        index = str(pre_index) + "C"

        for i, compiled_lists in enumerate(zip(self.comp_weights_list, self.comp_activation)):

            names = [
                "weights_compiled",
                "activations_compiled"
            ]

            for param_pair in zip(compiled_lists, names):
                running_mem_total = Filetools.save_to_file(
                param_pair[1] + index + str(post_start_index+i) + ".mem", 
                param_pair[0],
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

    def calculate_scale_w(self, target_list):

        max_val = np.amax([abs(x) for x in target_list])
        scale_val = 0

        while max_val <= 1.0:
            scale_val += 1
            max_val *= 2.0
            print(max_val)

        return scale_val-1

class Synapses_Fixed(Synapses):

    def __init__(self, n_neurons, output_dims, pstc_scale, decoders_list, encoders_list, n_activ_extra, n_output, radix_w, index, verbose=False):
       
        # TODO add support for multidimensional decoders here.

        scale_w = 5
        # Scale the weights by the post-synaptic scaling constant.
        decoders_list = decoders_list*pstc_scale

        super().__init__(self, n_neurons, output_dims, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, scale_w, verbose=False)

        # Compile the weights
        self.comp_weights_list, self.n_w = Compiler.compile_floats(self.weights, radix_w, verbose=verbose)

        self.save_params(index, floating=False)

class Synapses_Floating(Synapses):

    def __init__(self, pstc_scale, decoders_list, encoders_list, n_activ_extra, n_output, radix_w, minimum_val, pre_index, post_start_index, verbose=False):
        
        decoders_list = decoders_list * 1
        self.output_dims = np.shape(decoders_list)[0]

        n_neurons = np.shape(decoders_list)[-1]
        
        # Clip small values to reduce dynamic range and hence decrease required exponent bit depth.
        if minimum_val != 0:
            for i, weight_list in enumerate(decoders_list):
                decoders_list[i] = [Compiler.clip_value(x, minimum_val) for x in weight_list]

        scale_w = Compiler.determine_middle_exp(decoders_list.flatten())
        logger.info("Scale_w for %i: %i", post_start_index, scale_w)
        super().__init__(n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, n_output, radix_w, scale_w, verbose=False)

        # Compile the weights
        highest_exp = 0

        # Calculate the largest exponent needed for each weight list, to ensure the same bit depths across weights.
        for weight_list in self.weights:
            new_exp = Compiler.calculate_exponent_depth(weight_list)
            if new_exp > highest_exp:
                highest_exp = new_exp

        exp_limit = 1000
        self.comp_weights_list = []
        self.n_w_man = None
        self.n_w_exp = None
        for i, weight_list in enumerate(self.weights):
            comp_weights, n_w_man, n_w_exp = Compiler.compile_to_float(weight_list, self.radix_w, exp_limit, highest_exp, verbose=verbose)
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
