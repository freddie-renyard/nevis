import math
import numpy as np
from matplotlib import pyplot as plt, scale
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher

class Compiler:

    """ 
    This class contains methods for compiling the floating point model parameters to floating point or fixed
    point representations at the desired bit widths/precisions.

    Abbreviations:
        NAU - Neural Arithmetic Unit
    """

    @classmethod
    def compile_floats(cls, target_list, bin_places, verbose=False):

        """This method will compile a list of floating point parameters into binary numbers,
        represented as strings. The bit width for compilation is determined by the required
        level of fractional precision, as the numbers are represented as fixed point integers.

        Parameters
        ----------
        target_list: [float]
            The list to be compiled into fixed point integers.
        bin_places: int
            The number of bits desired for the fractional portion of the float.
            The number of bits needed for the non-fractional portion of the float is determined automatically.
        verbose: bool
            A flag which can be set to check each floating point value against it's
            compiled binary counterpart.


        Returns
        -------
        compiled_str: [str]
            The compiled list of parameters in binary as strings.
        total_bit_depth: int
            The bit depth of the parameters in the list, after consideration
            of the bits needed to store the non-fractional part of the float.
        """
        
        print('')
        print('Compiling list to desired fractional precision...')
        compiled_str = []
        int_bit_depth = 0

        max_value = np.amax([abs(x) for x in target_list])
        
        def num_split(x):
            
            sign = (x < 0)
            x = abs(x)

            int_part = math.floor(x)
            fractional_part = x - int_part

            overflow = (1 - (1 / (2 ** bin_places))) < fractional_part

            if overflow:
                int_part += 1
                fractional_part = 0

            return int_part, fractional_part, sign

        max_int, _, _ = num_split(max_value)

        int_bit_depth = math.ceil(math.log2(max_int + 1)) + 1

        print('Report: Whole Number Bit Depth - ' + str(int_bit_depth) + ' Fractional Bit Depth: ' + str(bin_places))
        print('Report: Total Bit Depth: ' + str(int_bit_depth + bin_places))

        i = 0
        for x in target_list:
            
            int_part, fractional_part, sign = num_split(x)
   
            bin_frac = cls.frac_to_truncated_bin(fractional_part, bin_places)
            # print(fractional_part, "binary fractional part: ", bin_frac)        
            bin_int = int("{0:032b}".format(int_part))
        
            final_number = str(bin_int) + bin_frac
            total_bit_depth = int_bit_depth + bin_places

            if len(str(final_number)) != total_bit_depth:
                final_number = "0" * (total_bit_depth - len(str(final_number))) + final_number
                
            if sign:
                is_negative_zero = (int(bin_frac) == 0) & (int(bin_int) == 0)
            else:
                is_negative_zero = False

            # Conversion to twos complement for the binary integers that require it
            
            if sign & ~is_negative_zero:
                
                flipped_number = ""
                for bit in final_number:
                    if bit == "0":
                        flipped_number += "1"
                    else:
                        flipped_number += "0"
                
                add_one = int(flipped_number, 2) + 1
                flipped_number = str(int("{0:032b}".format(add_one)))

                if len(flipped_number) != total_bit_depth:
                    flipped_number = flipped_number[0:total_bit_depth]
                
                final_number = flipped_number #self.twos_complementer(final_number, total_bit_depth)

            compiled_str.append(final_number)

            if len(final_number) != total_bit_depth:
                print("ERROR: Final Value is not equal to bit depth for neuron ", i)

            if verbose:
                print("Index: ", i, "Value: ", str(x), " --> ", str(final_number))
                i += 1
        
        return compiled_str, total_bit_depth

    @classmethod
    def compile_to_float(cls, target_list, radix_mantissa, exp_limit, verbose=False):
        """ This method compiles parameters to the proprietary floating point architecture
        specified in the encoder.

        Parameters
        ----------
        target_list: [float]
            The list to compile.
        radix_mantissa: int
            The desired radix of the mantissa.
        exp_limit
            The maximum magnitude of the exponent. Used to prevent the compiler from compiling
            extremely low values e.g. 1.0x2^-53. Default = 0 (no limit)
        verbose: bool
            Whether to output a detailed output of the process to the terminal.
        """

        print('')
        print('Compiling list to desired floating-point precision...')

        compiled_str = []

        # Normalised mantissa - add the sign bit to the total mantissa bit depth.
        mantissa_depth = radix_mantissa + 1

        # Compute the exponent bit depth needed to store the parameters
        max_value = np.amax([abs(x) for x in target_list])
        min_value = np.amin([abs(x) for x in target_list])

        def calculate_binary_exp(value):

            # Determine the binary exponent needed to store the value
            # with a -1.0 to ~1.0 normalised mantissa
            exponent = 0
            if value > 1.0:
                while value >= 1.0:
                    value /= 2
                    exponent += 1
            else:
                while value <= 1.0:
                    value *= 2
                    exponent += 1
                value /= 2
                exponent -= 1
            
            return exponent 
        
        # Calculate upper exponent - NB This method will hang if the largest value is 0.0...
        upper_exponent = calculate_binary_exp(max_value)

        # Check if the lowest value is a zero.
        if min_value < 1*2**-10:
            min_value = 1*2**-10

        lower_exponent = calculate_binary_exp(min_value)

        if upper_exponent > lower_exponent:
            largest_exp = upper_exponent
        else:
            largest_exp = lower_exponent

        # Add one for the sign
        exp_depth = math.ceil(math.log2(largest_exp + 1)) + 1 #max_exponent + 1

        # Iterate over every value
        i=0
        for x in target_list:
            
            # Calculate the exponent needed
            value = abs(x)
            man_sign = (x < 0)
            exponent_val = 0

            # Check if the value is zero.
            if value < 1*2**-10:
                value = 1*2**-10

            if value >= 1:
                while value >= 1:
                    value /= 2
                    exponent_val += 1
            else:
                while value < 1:
                    value *= 2
                    exponent_val -= 1
                value /= 2
                exponent_val += 1

            # Convert the mantissa to its binary string representation
            mantissa_bin, _, overflow = cls.frac_to_truncated_bin(value, radix_mantissa, man_sign)

            if not overflow:
                mantissa_bin = "0" + mantissa_bin

            if overflow and not man_sign:
                # Increase exponent, rounding the value up if the value is being rounded
                exponent_val += 1
                mantissa_bin = "0" + "1" + "0"*(radix_mantissa-1)

            # Convert the mantissa 2's complement if required.
            if man_sign and not overflow:
                mantissa_bin = cls.twos_complementer(mantissa_bin, mantissa_depth)

            # Convert the exponent to binary
            exp_sign = (exponent_val < 0)
            exponent_bin = str(int("{0:032b}".format(abs(exponent_val))))
            
            # This should be put into a function.
            if len(str(exponent_bin)) != exp_depth:
                # print("Length of exp_Detph", exp_depth, "Length of exponent_bin:", len(exponent_bin))
                exponent_bin = "0" * (exp_depth - len(exponent_bin)) + exponent_bin
            
            if exp_sign:
                exponent_bin = cls.twos_complementer(exponent_bin, exp_depth)

            if verbose:
                if man_sign:
                    display_val = value*-1
                else:
                    display_val = value
                print("Index:", i, "Value:", x, "Mantissa:", display_val, "Exponent:", exponent_val)
                print("Compiled Mantissa: ", mantissa_bin, "Compiled Exponent:", exponent_bin)
            
            # Concatenate the two entries together
            concat_result = mantissa_bin + exponent_bin

            # Add to the list
            compiled_str.append(concat_result)

            if verbose:
                i += 1

        return compiled_str, mantissa_depth, exp_depth

    @classmethod
    def clip_value(cls, value, threshold):
        # A function which rounds the values based on the exp_limit argument
        # Ensure input parameters are positive
        value_sign = (value < 0.0)
        value_abs = abs(value)
        threshold = abs(threshold)

        if value_abs < threshold:
            if value_sign:
                return threshold*-1
            else:
                return threshold 
        else:
            return value

    @staticmethod
    def frac_to_truncated_bin(fraction, bin_places, is_neg=False):
        
        """ NEW IMPLEMENTATION
        fraction = abs(fraction)
        if fraction > 1.0:
            fraction = 1.0
            print("[COMPILER WARNING]: Truncation in mantissa representation has occured")

        # Add an extra bit on to allow for compilation of the two's complement -max value 
        # condition. TODO Adapt the compile_to_float method to take advantage of this and
        # test it throrughly
        if is_neg:
            max_val = 2**bin_places + 1
        else: 
            max_val = 2**bin_places
        
        # Calculate the step for each threshold
        step = 1.0 / (max_val-1)

        # Compute thresholds (the ideal fixed point values for each given representation)
        thresholds = []
        for i in range(0, max_val):
            thresholds.append(i*step)
        
        # Compute the middle of each threshold (used for rounding)
        bin_frac = 0
        thresh_mid = ((thresholds[1] - thresholds[0]) / 2.0)

        for i in range(0, max_val):
            thresh_div = thresh_mid + thresholds[i]
            # Test where the fraction is in comparison to thresholds defined
            if fraction > thresholds[i+1]:
                pass
            elif fraction < thresh_div:
                # Round value down to lower value
                bin_frac = i
                break
            else:
                # Round value up to higher value
                bin_frac = i+1
                break
        
        """
        
        bin_frac = 0
        overflow_flag = False

        # This allows the compiler to exploit the extra value provided by negative two's complement 
        if is_neg:
            value_range = 2 ** bin_places
        else:
            value_range = 2 ** bin_places - 1

        for value in range(value_range):

            if bin_places == 0:
                break

            lower_bound = (value) * (1 / (2 ** bin_places))
            upper_bound = (value+1) * (1 / (2 ** bin_places))
            
            if lower_bound < fraction and fraction <= upper_bound:
                difference_1 = fraction - lower_bound
                difference_2 = upper_bound - fraction

                if difference_1 > difference_2:
                    bin_frac = value +1
                else:
                    bin_frac = value
            elif fraction > upper_bound:
                bin_frac = value +1
                
        step = 1.0 / (2 ** bin_places)
        if fraction > ((step)*((2 ** bin_places) - 1) + step/2.0):
            overflow_flag = True
        
        # Convert the integer to binary representation
        int_debug = bin_frac
        bin_frac = int("{0:032b}".format(bin_frac))
        
        # Pad with zeroes
        if len(str(bin_frac)) != bin_places and ~overflow_flag:
            bin_frac = "0" * (bin_places - len(str(bin_frac))) + str(bin_frac)

        return str(bin_frac), int_debug, overflow_flag

    @staticmethod
    def multiply_to_exp_lim(value, value_exp, exp_lim):
        """ This method converts a value with a given exponent into a 
        value with a exponent within the magnitude limit specified by
        the function parameters.

        Parameters:
        -----------
        value: float
            The mantissa for conversion
        value_exp: int
            The exponent of the mantissa specified
        exp_lim: int
            The magnitude limit for the exponent

        Returns:
        --------
        value: float
            The value, now within exponent limit
        new_exp: int
            The exponent, now within exponent limit
        """
        abs_exp = abs(value_exp)
        if value_exp < 0:
            new_exp = exp_lim*-1
            while abs_exp != exp_lim:
                abs_exp -= 1
                value /= 2
        else:
            new_exp = exp_lim
            while abs_exp != exp_lim:
                abs_exp -= 1
                value *= 2
        
        return value, new_exp

    @staticmethod
    def twos_complementer(bin_str, final_bit_depth):

        flipped_number = ""
        for bit in bin_str:
            if bit == "0":
                flipped_number += "1"
            else:
                flipped_number += "0"
        
        add_one = int(flipped_number, 2) + 1
        flipped_number = str(int("{0:032b}".format(add_one)))

        if len(flipped_number) != final_bit_depth:
            flipped_number = flipped_number[0:final_bit_depth]

        return flipped_number

    @staticmethod
    def determine_middle_exp(target_list):

        abs_list = [abs(x) for x in target_list]
        max_val = abs(np.amax(target_list))
        min_val = abs(np.amin(target_list))

        max_exponent = 0
        while max_val <= 1:
            max_val *= 2
            max_exponent += 1
            print(max_val)

        print(max_exponent)

        min_exponent = 0
        while min_val <= 1:
            min_val *= 2
            min_exponent += 1
            print(min_val)

        print("Max exponent:", -1*max_exponent, "Min exponent:", -1*min_exponent)
        return int((max_exponent + min_exponent)/2)

    @staticmethod
    def calculate_refractory_params(refractory, timestep):

        """Calculates the appropriate hardware parameters for the refractory period specified.

        Parameters
        ----------
        refractory : float
            The refractory period in the context of the original model's timestep.
        timestep : float
            The timestep of the original model.

        Returns
        -------
        period: int
            The new refractory period in context of the hardware implementation of the LIF spiking behaviour.
        bit_width: int
            The hardware bitwidth needed to store the value and produce the appropriate overflow
            for hardware refractory period behaviour.
        """
        
        period = refractory / timestep
        bit_width = math.ceil(math.log2(period+1))
        period = 2 ** bit_width - period - 1

        return int(period), int(bit_width)

    @staticmethod
    def calculate_t_rc_shift(scaled_trc):
        """ This method takes the scaled T_RC value (t_rc / dt) from the Nengo model,
        and returns the number of bits that will need to be shifted to realise the division
        in hardware.
        """
        return int(math.log2(scaled_trc))

    @classmethod
    def test_int_bin_conversion(cls):
        """ A method for testing the integer conversion function in the compiler.
        """
        
        resolution = 10000
        test_list = [x/resolution for x in range(0, resolution)]
        results = [0]*len(test_list)
        results_debug = [0]*len(test_list)
        flags_debug = [False]*len(test_list)
        
        bin_places = 4


        for i in range(len(test_list)):
            results[i], results_debug[i], flags_debug[i] = cls.frac_to_truncated_bin(test_list[i], bin_places, is_neg=True)

        # Calculate RMSE for the approximations against their floating point representations
        rmse_sum = 0
        scale_factor = 2**(bin_places)
        for x,y in zip(test_list, results_debug):
            
            y = float(y) / scale_factor
            print(x,y)
            rmse_sum += (y - x)**2
        
        from math import sqrt
        rmse = sqrt(rmse_sum / len(test_list))

        print("RMSE for this compiler function is", rmse_sum)
        
        print([x for x in zip(test_list, flags_debug)])
        plt.plot(test_list, results)
        for x in range(scale_factor):
            plt.axvline(x=float(x)/scale_factor, color='r', linestyle='--')
        plt.title("Unsigned fixed-point interger representation against input value")
        plt.xlabel("Value")
        plt.ylabel("Binary number")
        plt.show()
    
    @classmethod
    def test_float_compiler(cls):

        mantissa = 3

        resolution = 1000 # How many steps there are for each of the positive and negative portions of the evaluation
        gain = 2 # The dynamic range tested
        test_inputs = [x/resolution for x in range(-resolution, resolution+1)]
        test_inputs = [x*gain for x in test_inputs]

        concat_numbers, mantissa_depth, exp_depth = cls.compile_to_float(test_inputs, mantissa, exp_limit=0.0, verbose=True)

        # Deconcatenate the data
        exp_bins = [x[mantissa_depth:] for x in concat_numbers]
        mantissa_bins = [x[:mantissa_depth] for x in concat_numbers]

        def abs_and_sign(binaries):
            signs = []
            abs_vals = []
            depth = len(binaries[0])
            for binary in binaries:
                if binary[0] == "1":
                    abs_val = cls.twos_complementer(binary, depth)
                    abs_vals.append(abs_val)
                    signs.append(-1)
                else:
                    abs_vals.append(binary)
                    signs.append(1)

            final_ints = [int(x, 2) * y for x, y in zip(abs_vals, signs)] 
            
            return final_ints

        # Convert mantissae to integers
        man_ints = abs_and_sign(mantissa_bins)
        man_ints = [x / (2**mantissa) for x in man_ints]

        # Convert exponents to integers
        exp_ints = abs_and_sign(exp_bins)

        # Scale mantissae by exponents
        final_vals = [x*(2**y) for x,y in zip(man_ints, exp_ints)]

        # Examine the final list for errors 
        for i in range(0, len(final_vals)):
            # If the value is very different to the input value
            error = abs(final_vals[i] - test_inputs[i])
            if error > 10:
                print("ERROR:", 
                    "Input Value:", round(test_inputs[i], 2), 
                    "Decoded Value:", round(final_vals[i], 2), 
                    "Mantissa", mantissa_bins[i], 
                    "Exponent:", exp_bins[i],
                    "Error factor:", (test_inputs[i] / final_vals[i])
                )
        
        import mplcursors
        plt.plot(test_inputs, final_vals)
        plt.plot(final_vals, final_vals)
        plt.xlabel("Input values")
        plt.ylabel("Floating point representation")
        plt.legend(["Mantissa Depth: " + str(mantissa+1) +  "\nExponent Depth: " + str(len(exp_bins[0]))])
        plt.title("Decoded compiled floating point values against a test input")
        mplcursors.cursor(hover=True)
        plt.show()

#com = Compiler()
#com.test_float_compiler()
"""
NOTES

REDESIGN OF THE ARITHMETIC ARCHITECTURE

Improvements to the compilation process would include defining a different method
to allow direct specification of bit depth, rather than deriving it from maximum
values. Under the current fixed-point hardware, this would present large accuracy 
issues, as the lists have large ranges (0.001-400). This could be alleviated by
developing a proprietary floating point representation (NOT IEEE 734!!) and
accompanying hardware architecture.

This would make the logic more complex at the benefit of saving memory, which
will almost certainly be the limiting factor for this system. Algorithm ideas:
1. calculate dynamic range aka if 230 or 0.5 -> 2^7 -2^-1 => dynamic range of 8
    therefore bits needed $clog2(8) = 3
2. User-specified significand bit depth eg 5 -> 4 number bits + 1 sign bit 
3. Total bit depth = 5 + 3 = 8
4. Perform appropriate binary conversion/truncation algorithms
5. Output list and parameters: n_significand, n_exponent

Only the computation modules of the system would need full redesign, the memory
FSMs could do with reparameterising although n_g = n_significand + n_exponent.
The data can be deconcatenated in the MAC/NAU.


"""


