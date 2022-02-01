import math
import numpy as np
from matplotlib import pyplot as plt, scale
import logging
from math import sqrt

logger2 = logging.getLogger(__name__)

class Compiler:

    """ 
    This class contains methods for compiling the floating point model parameters to floating point or fixed
    point representations at the desired bit widths/precisions.

    Abbreviations:
        NAU - Neural Arithmetic Unit
    """

    @classmethod
    def compile_to_fixed(cls, target_list, bin_places, verbose=False):

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

        compiled_str = []
        int_bit_depth = 0

        # Find the maximum integer value that will need to be
        # represented in the list, and the number of bits needed
        # to store it
        max_value = np.amax([abs(x) for x in target_list])
        max_int, _, _ = cls.num_split(max_value, bin_places)
        int_bit_depth = math.ceil(math.log2(max_int + 1)) + 1

        logger2.info('Report: Whole Number Bit Depth: %4f, Fractional Bit Depth: %4f', int_bit_depth, bin_places)
        logger2.info('Report: Total Bit Depth: %s', str(int_bit_depth + bin_places))

        for i, x in enumerate(target_list):
            
            # Split the number apart into its unsigned integer and fractional parts, plus sign
            int_part, fractional_part, sign = cls.num_split(x, bin_places)

            # Convert the fractional part to binary string
            bin_frac, _ , _ = cls.frac_to_truncated_bin(fractional_part, bin_places)   

            # Convert the integer part to binary  
            bin_int = int("{0:032b}".format(int_part))

            # Combine the integer and fractional part into a full unsigned binary string
            final_number = str(bin_int) + bin_frac
            total_bit_depth = int_bit_depth + bin_places

            # Zero pad numbers which are too short for the memory width
            if len(str(final_number)) != total_bit_depth:
                final_number = "0" * (total_bit_depth - len(str(final_number))) + final_number
            
            # Catch the negative zero representation
            if sign:
                is_negative_zero = (int(bin_frac) == 0) & (int(bin_int) == 0)
            else:
                is_negative_zero = False

            # Conversion to twos complement for the binary integers that represent negative 
            # numbers
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
                
                final_number = flipped_number

            compiled_str.append(final_number)

            # Logging and debug
            if len(final_number) != total_bit_depth:
                logger2.error("ERROR: Final Value is not equal to bit depth for neuron %i", i)

            if verbose:
                logger2.info("INFO: Index: %i, Value: %.5f, --> %s", i, x, str(final_number))
        
        return compiled_str, total_bit_depth

    @classmethod
    def compile_to_float(cls, target_list, radix_mantissa, force_exp_depth=None, verbose=False):
        """ This method compiles parameters to the proprietary floating point architecture
        specified in the encoder.

        Parameters
        ----------
        target_list: [float]
            The list to compile.
        radix_mantissa: int
            The desired radix of the mantissa.
        force_exp_depth : 
            The desired depth of the exponent, for compiling multidimensional decoders.
        verbose: bool
            Whether to print a detailed output of the process to the terminal.
        """

        compiled_str = []

        # Normalised mantissa - add the sign bit to the total mantissa bit depth.
        mantissa_depth = radix_mantissa + 1

        # If a exponent depth hasn't been manually set, compute it
        if force_exp_depth is None:
            exp_depth = cls.calculate_exponent_depth(target_list)
        else:
            exp_depth = force_exp_depth
        
        # Iterate over every value in the list to be compiled.
        for i, x in enumerate(target_list):
            
            # Normalise the number to within 1 and -1, and simultaneously
            # compute the exponent of the value (mantissa x 2 ^ exponent)
            value = abs(x)
            man_sign = (x < 0)
            exponent_val = 0

            if value >= 1.0:
                while value >= 1.0:
                    value /= 2
                    exponent_val += 1
            else:
                while value < 1.0:
                    value *= 2
                    exponent_val -= 1
                value /= 2
                exponent_val += 1

            # Convert the mantissa to its unsigned binary string representation
            mantissa_bin, _, overflow = cls.frac_to_truncated_bin(value, radix_mantissa, man_sign)

            # Add a positive sign bit if the value hasn't overflowed
            if not overflow:
                mantissa_bin = "0" + mantissa_bin

            if overflow and not man_sign:
                # Increase exponent, rounding the value up if the value is being rounded
                # TODO Add a catch case here for the situation where depth of the overflowed exponent
                # is greater than allocated bit depth.
                exponent_val += 1
                mantissa_bin = "0" + "1" + "0"*(radix_mantissa-1)
                logger2.warning("An overflow has occured. This can lead to unexpected float compiler behaviour.")

            # Convert the mantissa to 2's complement if value is negative.
            if man_sign and not overflow:
                mantissa_bin = cls.twos_complementer(mantissa_bin, mantissa_depth)

            # Convert the exponent to it's binary string representation
            exp_sign = (exponent_val < 0)
            exponent_bin = str(int("{0:032b}".format(abs(exponent_val))))
            
            # Zero pad the exponent out if the exponent is not the same length as the alloted memory entry.
            if len(str(exponent_bin)) != exp_depth:
                exponent_bin = "0" * (exp_depth - len(exponent_bin)) + exponent_bin
            
            # Convert the exponent to 2's complement if value is negative.
            if exp_sign:
                exponent_bin = cls.twos_complementer(exponent_bin, exp_depth)

            # Log the values for debug
            if verbose:
                if man_sign:
                    display_val = value*-1
                else:
                    display_val = value
                logger2.info("Index: %i, Value: %.5f, Mantissa: %.5f, Exponent: %.5f", i, x, display_val, exponent_val)
                logger2.info("Compiled Mantissa: %s, Compiled Exponent: %s", mantissa_bin, exponent_bin)
                i += 1
            
            # Concatenate the two entries together and add to the list to compile to memory file.
            concat_result = mantissa_bin + exponent_bin
            compiled_str.append(concat_result)
                
        return compiled_str, mantissa_depth, exp_depth

    @classmethod 
    def calculate_exponent_depth(cls, target_list):
        """Compute the exponent bit depth needed to store the parameters in a target list.
        """

        max_value = np.amax([abs(x) for x in target_list])
        min_value = np.amin([abs(x) for x in target_list]) 

        # Check if the lowest or highest value are zero.
        if min_value < 1*10**-50:
            min_value = 1*10**-50

        if max_value < 1*10**-50:
            max_value = 1*10**-50
        
        # Calculate upper and lower exponents
        upper_exponent = cls.calculate_binary_exp(max_value)
        lower_exponent = cls.calculate_binary_exp(min_value)

        # Return the maximum exponent magnitude
        if upper_exponent > lower_exponent:
            largest_exp = upper_exponent
        else:
            largest_exp = lower_exponent
        
        # Return the number of bits needed to store the value.
        # Add one for the sign.
        return math.ceil(math.log2(largest_exp + 1)) + 1

    @staticmethod
    def calculate_binary_exp(value):
        """Determine the binary exponent needed to store the value
        with a -1.0 to ~1.0 normalised mantissa.
        """

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
    
    @classmethod
    def clip_value(cls, value, threshold):
        """A method which rounds a value based on a threshold.
        """

        # Ensure the input parameters are positive
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
    def num_split(x, bin_places):
        # Split a given number into its unsigned fractional and integer parts, along with its sign

        sign = (x < 0)
        x = abs(x)

        int_part = math.floor(x)
        fractional_part = x - int_part
        
        # Test if the smallest rounded value has rounded over the maximum value
        # that the binary number of the given depth can represent
        overflow = (1 - (1 / (2 ** bin_places))) < fractional_part

        if overflow:
            int_part += 1
            fractional_part = 0

        return int_part, fractional_part, sign

    @staticmethod
    def frac_to_truncated_bin(fraction, bin_places, is_neg=False):
        """Compute the rounded binary fraction that can be represented
        in the given number of binary places. 
        """
        
        bin_frac = 0
        overflow_flag = False

        # This allows the compiler to exploit the extra value provided by a
        # negative two's complement representation.
        if is_neg:
            value_range = 2 ** bin_places
        else:
            value_range = 2 ** bin_places - 1

        # Calculate the desired amount of fractional value precision.
        # This is done by rounding the number's magnitude to the closest
        # binary string representation.
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
            
        # Check overflow to ensure to catch potential positive number misrepresentation.
        step = 1.0 / (2 ** bin_places)
        if fraction > ((step)*((2 ** bin_places) - 1) + step/2.0):
            overflow_flag = True
        
        # Convert the integer to binary representation
        int_debug = bin_frac
        bin_frac = int("{0:032b}".format(bin_frac))
        
        # Pad with zeroes if needed
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
        """Converts a binary number, represented as a string, 
        to its two's complement representation.
        """

        flipped_number = ""

        # First take the one's complement by inverting all the bits
        for bit in bin_str:
            if bit == "0":
                flipped_number += "1"
            else:
                flipped_number += "0"
        
        add_one = int(flipped_number, 2) + 1
        flipped_number = str(int("{0:032b}".format(add_one)))

        # Truncate the flipped number if it is outside of the specified  
        # binary string length
        if len(flipped_number) != final_bit_depth:
            flipped_number = flipped_number[0:final_bit_depth]

        return flipped_number

    @staticmethod
    def determine_middle_exp(target_list):
        """Determine the exponent needed to represent a given 
        middle value of a list, which is defined as the mean
        of the exponents needed to represent the largest and 
        smallest values.
        """
        max_val = np.amax(np.abs(target_list))
        min_val = np.amin(np.abs(target_list))

        max_exponent = math.ceil(math.log2(max_val))
        min_exponent = math.ceil(math.log2(min_val))
        
        # Return the mean average of the max and minimum exponents.
        return abs(int((max_exponent + min_exponent)/2))

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
            rmse_sum += (y - x)**2
        
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
        """A method for evaluation of the floating point compiler.
        """

        mantissa = 3

        resolution = 1000 # How many steps there are for each of the positive and negative portions of the evaluation
        gain = 2 # The dynamic range tested
        test_inputs = [x/resolution for x in range(-resolution, resolution+1)]
        test_inputs = [x*gain for x in test_inputs if x != 0.0]

        # Compile the parameters.
        concat_numbers, mantissa_depth, exp_depth = cls.compile_to_float(test_inputs, mantissa, force_exp_depth=8, verbose=True)

        # Deconcatenate the data
        exp_bins = [x[mantissa_depth:] for x in concat_numbers]
        mantissa_bins = [x[:mantissa_depth] for x in concat_numbers]

        # Convert mantissae to integers
        man_ints = cls.abs_and_sign(mantissa_bins)
        man_ints = [x / (2**mantissa) for x in man_ints]

        # Convert exponents to integers
        exp_ints = cls.abs_and_sign(exp_bins)

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

    @classmethod
    def abs_and_sign(cls, binaries):
        """Convert list of binary values (represented as strings)
        to their corresponding signed integer representation.
        """
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

Compiler.test_float_compiler()

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


