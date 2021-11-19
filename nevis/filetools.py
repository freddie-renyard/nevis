import os
import logging

logger = logging.getLogger()

class Filetools:

    @staticmethod
    def combine_binary_params(list_0, list_1):

        """ This method concatenates two strings together and returns the list.
        In the context of the overall compiler, this method is used for 
        combining the binary gain and bias lists together to prepare for 
        memory storage.

        Parameters
        ----------
        list_0: [str]
            The first list to be concatenated (will appear as the LSBs of each
            concatenated entry)
        list_1: [str]
            The second list to be concatenated (will appear as the MSBs of each
            concatenated entry)

        Returns
        -------
        combined: [str]
            The concatenated inputs.
        """

        combined = [(x + y) for x, y in zip(list_0, list_1)]

        return combined

    @staticmethod
    def purge_directory(dir_path):
        """ Purges all files in a directory. Does not work for subdirectories.
        """
        for f in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, f))

    @classmethod
    def save_to_file(cls, filename, target_list, running_mem_total=0):
        """ Saves a specified list to a memory file in the working directory with entry comments.
        
        Parameters
        ----------
        filename: str
            The desired filename. Please supply with the '.mem' file type
        target_list: [str]
            The binary parameters supplied as a list of strings.
        running_mem_total: int
            A running total of the total number of bits a model uses.

        Returns
        -------
        running_mem_total: int
            Adds the number of bits compiled to a running total, which can be passed.
        """

        path = cls.open_cache(filename)
        file = open(path, "w")

        i = 0
        total_bits = len(target_list[0]) * len(target_list)

        division_val = 1.0
        unit = ' bits'
        if total_bits >= 1000:
            division_val = 1000.0
            unit = 'kb'
        
        logger.info("INFO: Memory usage for %s: %s", filename, (str(int(total_bits/division_val)) + unit))
        running_mem_total += total_bits

        for element in target_list:
            file.write("//" + "Entry " + str(i) + '\n')
            file.write(element + " " + '\n')
            i += 1
        
        return running_mem_total

    @staticmethod
    def report_memory_usage(bits_used):
        logger.info("INFO: Total number of bits used for compiled parameters: %s kb", str(bits_used/1000.0))

    @staticmethod
    def open_cache(filename):
        # Obtain path to cache
        path = os.path.realpath(__file__)
        dir = os.path.dirname(path) + "/file_cache"

        # Create the cache if it does not exist
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        return str(dir) + "/" + filename
    
    @classmethod
    def compile_and_save_header(cls, filename, full_model, global_params):
        """Compile the parameters in the objects specified
        into a Verilog header file to make transfer of parameters to hardware easier. This method
        will also save the compiled verilog header (.vh) file into the source directory
        under the filename specified. 

        Parameters
        ----------
        filename: str
            The specified filename for the header. Please supply filename with .vh extension.
        full_model: [<instances of compiled classes>]
            The model as a list of compiled instances of model components.
        global_params: [int]
            Parameters which do not change for each module in hardware and therefore are
            only compiled once: at present, only refractory period and its bit depth are 
            compiled this way.

        Returns
        -------
        None.
        """

        path = cls.open_cache(filename)
        file = open(path, "w")
        pop_count = 0
        pop_names = ['A', 'B', 'C', 'D']

        for sec in full_model:
            report_tag = "Population "
            class_name = sec.__class__.__name__
            if class_name == 'Encoder':
                index = pop_names[pop_count]
                report_tag += index + " Encoder:"
                print("\nENSEMBLE", index, ": INTEGER ENCODER NEURONS\n")
                file.write(("// Population " + index + ' Params' + '\n'))
                file.write(('parameter ' + 'N_NEURON_' + index + ' = ' + str(sec.n_neurons) + ',' + '\n'))
                # X/Incoming activation Params
                file.write(('N_X_' + index + ' = ' + str(sec.n_x) + ',' + '\n'))
                file.write(('RADIX_X_' + index + ' = ' + str(sec.radix_x) + ',' + '\n'))
                # Gain Params
                file.write(('N_G_' + index + ' = ' + str(sec.n_g) + ',' + '\n'))
                file.write(('RADIX_G_' + index + ' = ' + str(sec.radix_g) + ',' + '\n'))
                # Bias Params
                file.write(('N_B_' + index + ' = ' + str(sec.n_b) + ',' + '\n'))
                file.write(('RADIX_B_' + index + ' = ' + str(sec.radix_b) + ',' + '\n'))
                # Output value to NAU param
                file.write(('N_DV_POST_' + index + ' = ' + str(sec.n_dv_post) + ';' + '\n'))
                file.write('\n') 
            elif class_name == 'Synapses':
                index = pop_names[pop_count]
                report_tag += index + " Synapses:"
                file.write(("// Population " + index + ' Synaptic Params' + '\n'))
                # Weight Params
                file.write(('parameter ' + 'N_WEIGHT_' + index + ' = ' + str(sec.n_w) + ',' + '\n'))
                file.write(('SCALE_W_' + index + ' = ' + str(sec.scale_w) + ',' + '\n'))
                file.write(('N_WEIGHT_EXP_' + index + ' = ' + str(0) + ',' + '\n'))
                # Activation Params
                file.write(('N_ACTIV_EXTRA_' + index + ' = ' + str(sec.n_activ_extra) + ',' + '\n'))
                # Synaptic time constant
                file.write(('PSTC_SHIFT_' + index + ' = ' + str(sec.pstc_shift) + ';' + '\n'))
                # file.write(('ACTIV_L_SHIFT_' + index + ' = ' + str(sec.activ_l_shift) + ';' + '\n')) Obsolete
                file.write('\n')
                pop_count += 1
            elif class_name == 'Synapses_Floating':
                index = pop_names[pop_count]
                report_tag += index + " Synapses (Floating):"
                file.write(("// Population " + index + ' Synaptic Params' + '\n'))
                # Weight Params
                file.write(('parameter ' + 'N_WEIGHT_' + index + ' = ' + str(sec.n_w) + ',' + '\n'))
                file.write(('SCALE_W_' + index + ' = ' + str(sec.scale_w) + ',' + '\n'))
                file.write(('N_WEIGHT_EXP_' + index + ' = ' + str(sec.n_w_exp) + ',' + '\n'))
                # Activation Params
                file.write(('N_ACTIV_EXTRA_' + index + ' = ' + str(sec.n_activ_extra) + ',' + '\n'))
                # Synaptic time constant
                file.write(('PSTC_SHIFT_' + index + ' = ' + str(sec.pstc_shift) + ';' + '\n'))
                # file.write(('ACTIV_L_SHIFT_' + index + ' = ' + str(sec.activ_l_shift) + ';' + '\n')) Obsolete
                file.write('\n')
                pop_count += 1
            elif class_name == 'Encoder_Floating':
                index = pop_names[pop_count]
                report_tag += index + " Encoder:"
                file.write(("// Population " + index + ' Params' + '\n'))
                file.write(('parameter ' + 'N_NEURON_' + index + ' = ' + str(sec.n_neurons) + ',' + '\n'))
                # X/Incoming activation Params
                file.write(('N_X_' + index + ' = ' + str(sec.n_x) + ',' + '\n'))
                file.write(('RADIX_X_' + index + ' = ' + str(sec.radix_x) + ',' + '\n'))
                # Gain Params
                file.write(('N_G_MAN_' + index + ' = ' + str(sec.n_g_mantissa) + ',' + '\n'))
                file.write(('N_G_EXP_' + index + ' = ' + str(sec.n_g_exponent) + ',' + '\n'))
                # Bias Params
                file.write(('N_B_MAN_' + index + ' = ' + str(sec.n_b_mantissa) + ',' + '\n'))
                file.write(('N_B_EXP_' + index + ' = ' + str(sec.n_b_exponent) + ',' + '\n'))
                # Output value to NAU param
                file.write(('N_DV_POST_' + index + ' = ' + str(sec.n_dv_post) + ';' + '\n'))
                file.write('\n')

        file.write(("// Global Synaptic Params" + '\n'))
        file.write(('parameter N_R = ' + str(global_params[0])  + ',' + '\n'))
        file.write(('REF_VALUE = ' + str(global_params[1])  + ',' + '\n'))
        file.write(('T_RC_SHIFT = ' + str(global_params[2])  + ';' + '\n'))