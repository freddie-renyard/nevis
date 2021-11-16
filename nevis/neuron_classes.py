from proto_nevis.memory_compiler import Compiler
import math
import numpy as np

from proto_nevis.filetools import Filetools
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

class Encoder:
    
    def __init__(self, n_neurons, gain_list, encoder_list, bias_list, t_rc, n_x, radix_x, n_dv_post):
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
        # Multiply the gains by their respective encoder values and divide by RC constant.
        self.eg_trc_list = [(x * y) / t_rc for x, y in zip(gain_list, encoder_list)]
        self.b_trc_list = [x / t_rc for x in bias_list]

        # Range of x is confined to 0.99... to -1 (Q0.x)
        self.n_x = n_x
        self.radix_x = radix_x

        self.n_neurons = n_neurons

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

    def save_params(self, index, n_r, running_mem_total=0):

        filename = "encoder_compiled" + str(index) + ".mem"
        print("Saving gain and bias to binary .mem file as", filename + "............", end='')
        combined = Filetools.combine_binary_params(self.comp_gain_list, self.comp_bias_list)
        running_mem_total = Filetools.save_to_file(
            filename, 
            combined,
            running_mem_total
        )

        filename = "nau_compiled" + str(index) + ".mem"
        print("Saving NAU start parameters to binary .mem file as", filename + "............", end='')
        combined = self.compile_nau_start_params(n_activ=self.n_dv_post, n_ref=n_r, n_neuron=self.n_neurons)
        running_mem_total = Filetools.save_to_file(
            filename, 
            combined,
            running_mem_total
        )

        return running_mem_total

class Encoder_Fixed(Encoder):
    def __init__(self, n_neurons, gain_list, encoder_list, bias_list, t_rc, n_x, radix_x, radix_g, radix_b, n_dv_post, verbose=False):
        self.radix_g = radix_g
        self.radix_b = radix_b
        super().__init__(self, n_neurons, gain_list, encoder_list, bias_list, t_rc, n_x, radix_x, n_dv_post)

        # Compile the gain and bias into seperate lists.
        self.comp_gain_list, self.n_g = Compiler.compile_floats(self.eg_trc_list, self.radix_g, verbose=verbose)
        self.comp_bias_list, self.n_b = Compiler.compile_floats(self.b_trc_list, self.radix_b, verbose=verbose)

class Encoder_Floating(Encoder):
    def __init__(self, n_neurons, gain_list, encoder_list, bias_list, t_rc, n_x, radix_x, radix_g, radix_b, n_dv_post, verbose=False):

        self.radix_g_mantissa = radix_g
        self.radix_b_mantissa = radix_b

        super().__init__(n_neurons, gain_list, encoder_list, bias_list, t_rc, n_x, radix_x, n_dv_post)

        # Compile the gain and bias into seperate lists.
        exp_limit = 15
        self.comp_gain_list, self.n_g_mantissa, self.n_g_exponent = Compiler.compile_to_float(self.eg_trc_list, self.radix_g_mantissa, exp_limit, verbose=verbose)
        self.comp_bias_list, self.n_b_mantissa, self.n_b_exponent = Compiler.compile_to_float(self.b_trc_list, self.radix_b_mantissa, exp_limit, verbose=verbose)

class Synapses:

    def __init__(self, n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, scale_w, verbose=False):
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
        decoders_list: [float]
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

        # This value is usually compiled in the Verilog, but is needed to compile
        # the starting memory files.
        self.n_activ = 1 + self.radix_w + self.scale_w
        
        # Calculate the number of bits to shift by to implement the post_synaptic filter.
        n_value = 1 / pstc_scale
        self.pstc_shift = int(math.log2(n_value))
        
        self.n_neurons = n_neurons
        self.n_activ_extra = n_activ_extra

        # Multiply weights by scale factor
        scale_factor_w = 2 ** self.scale_w
        self.weights = [x*scale_factor_w for x in decoders_list]
        
        # Check that the magnitude of the weights are less than 1.
        for value in self.weights:
            if abs(value) > 1:
                print("ERROR: Weights are too large for hardware.")
        
        self.comp_activation = self.compile_activations(n_activ=self.n_activ, n_neuron=1)
        self.comp_trait_bits = self.compile_traits(encoders_list)

    def compile_traits(self, encoders):
        """ Compiles the encoders into a list of 'trait' bits, which are used in
        the synaptic module for performing the dot product with the weights to 
        save memory usage. These bits are 0 for a neuron which reponds with a 
        positive weight, and 1 for a neuron which responds with a negative weight.

        Parameters
        ----------
        encoders: [Any]
            The list of encoders to be compiled.

        Returns
        -------
        comp_traits: [str]
            The compiled 'trait' bits in a list, ready for saving to .mem file.
        """

        comp_traits = []
        for element in encoders:
            if element == 1:
                comp_traits.append('0')
            elif element == -1:
                comp_traits.append('1')
            else:
                comp_traits.append('ERROR')
        return comp_traits
    
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

    def save_params(self, index, running_mem_total=0):

        params_to_save = [
            self.comp_weights_list,
            self.comp_activation,
            self.comp_trait_bits
        ]

        names = [
            "weights_compiled",
            "activations_compiled",
            "trait_bits_compiled"
        ]

        for param_pair in zip(params_to_save, names):
            running_mem_total = Filetools.save_to_file(
            param_pair[1] + str(index) + ".mem", 
            param_pair[0],
            running_mem_total
            )

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

    def __init__(self, n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, verbose=False):
       
        scale_w = 5
        # Scale the weights by the post-synaptic scaling constant.
        decoders_list = [x*pstc_scale for x in decoders_list]

        super().__init__(self, n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, scale_w, verbose=False)

        # Compile the weights
        self.comp_weights_list, self.n_w = Compiler.compile_floats(self.weights, radix_w, verbose=verbose)

class Synapses_Floating(Synapses):

    def __init__(self, n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, minimum_val, verbose=False):
        
        # Scale the weights by the post-synaptic scaling constant.
        decoders_list = [x*pstc_scale for x in decoders_list]

        # Clip small values to reduce dynamic range and hence decrease required exponent bit depth.
        if minimum_val != 0:
            decoders_list = [Compiler.clip_value(x, minimum_val) for x in decoders_list]

        scale_w = Compiler.determine_middle_exp(decoders_list)
        super().__init__(n_neurons, pstc_scale, decoders_list, encoders_list, n_activ_extra, radix_w, scale_w, verbose=False)

        # Compile the weights.
        exp_limit = 15
        self.comp_weights_list, self.n_w_man, self.n_w_exp = Compiler.compile_to_float(self.weights, self.radix_w, exp_limit, verbose=verbose)
        self.n_w = self.n_w_exp + self.n_w_man
