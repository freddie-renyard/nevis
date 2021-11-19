import os
import logging

class Filetools:

    @staticmethod
    def get_logger(file_name, filemode='w'):
        # Obtain raw filename
        name_components = file_name.split(".")
        filepath = "nevis/logs/" + name_components[-1] + ".log"

        # Save the log file
        logging.basicConfig(
            filename=filepath,
            format='%(asctime)s %(message)s',
            filemode=filemode
        )
        return logging.getLogger(file_name)


    @staticmethod
    def combine_binary_params(list_0, list_1, verbose=False):

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
        verbose: bool
            A flag that can be set to print the combined list and each of the 
            entries to check desired results.

        Returns
        -------
        combined: [str]
            The concatenated inputs.
        """

        combined = [(x + y) for x, y in zip(list_0, list_1)]

        if verbose:
            for i in range(len(combined)):
                print(combined[i], list_0[i], list_1[i])

        return combined

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
        
        print(("Memory usage for " + filename + ": " + str(int(total_bits/division_val)) + unit) + "\n")
        running_mem_total += total_bits

        for element in target_list:
            file.write("//" + "Entry " + str(i) + '\n')
            file.write(element + " " + '\n')
            i += 1
        
        return running_mem_total

    @staticmethod
    def report_memory_usage(bits_used):
        print(("Total number of bits used for compiled parameters: " + str(bits_used/1000.0) + "kb"))
        print("\n/-----------------------------/\n")

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

        print("\n/---------FINAL REPORT----------/")

        for sec in full_model:
            report_tag = "Population "
            class_name = sec.__class__.__name__
            if class_name == 'Encoder':
                index = pop_names[pop_count]
                report_tag += index + " Encoder:"
                print("\nENSEMBLE", index, ": INTEGER ENCODER NEURONS\n")
                file.write(("// Population " + index + ' Params' + '\n'))
                file.write(('parameter ' + 'N_NEURON_' + index + ' = ' + str(sec.n_neurons) + ',' + '\n'))
                print(report_tag, "Number of neurons: ", sec.n_neurons)
                # X/Incoming activation Params
                file.write(('N_X_' + index + ' = ' + str(sec.n_x) + ',' + '\n'))
                print(report_tag, "Input bit depth: ", sec.n_x)
                file.write(('RADIX_X_' + index + ' = ' + str(sec.radix_x) + ',' + '\n'))
                print(report_tag, "Input radix: ", sec.radix_x)
                # Gain Params
                file.write(('N_G_' + index + ' = ' + str(sec.n_g) + ',' + '\n'))
                print(report_tag, "Gain bit depth: ", sec.n_g)
                file.write(('RADIX_G_' + index + ' = ' + str(sec.radix_g) + ',' + '\n'))
                print(report_tag, "Gain radix: ", sec.radix_g)
                # Bias Params
                file.write(('N_B_' + index + ' = ' + str(sec.n_b) + ',' + '\n'))
                print(report_tag, "Bias bit depth: ", sec.n_b)
                file.write(('RADIX_B_' + index + ' = ' + str(sec.radix_b) + ',' + '\n'))
                print(report_tag, "Bias radix: ", sec.radix_b)
                # Output value to NAU param
                file.write(('N_DV_POST_' + index + ' = ' + str(sec.n_dv_post) + ';' + '\n'))
                print(report_tag, "Bit depth of datapath to the NAU: ", sec.n_dv_post)
                file.write('\n') 
            elif class_name == 'Synapses':
                index = pop_names[pop_count]
                report_tag += index + " Synapses:"
                file.write(("// Population " + index + ' Synaptic Params' + '\n'))
                print("\nENSEMBLE", index, ": SYNAPSES\n")
                # Weight Params
                file.write(('parameter ' + 'N_WEIGHT_' + index + ' = ' + str(sec.n_w) + ',' + '\n'))
                print(report_tag, "Weight bit depth: ", sec.n_w)
                file.write(('SCALE_W_' + index + ' = ' + str(sec.scale_w) + ',' + '\n'))
                print(report_tag, "Weight scale value: ", sec.scale_w)
                file.write(('N_WEIGHT_EXP_' + index + ' = ' + str(0) + ',' + '\n'))
                # Activation Params
                file.write(('N_ACTIV_EXTRA_' + index + ' = ' + str(sec.n_activ_extra) + ',' + '\n'))
                print(report_tag, "Extra bit depth in weight path: ", sec.n_activ_extra)
                # Synaptic time constant
                file.write(('PSTC_SHIFT_' + index + ' = ' + str(sec.pstc_shift) + ';' + '\n'))
                print(report_tag, "Post-synaptic time constant shift: ", sec.pstc_shift)
                # file.write(('ACTIV_L_SHIFT_' + index + ' = ' + str(sec.activ_l_shift) + ';' + '\n')) Obsolete
                file.write('\n')
                pop_count += 1
            elif class_name == 'Synapses_Floating':
                index = pop_names[pop_count]
                report_tag += index + " Synapses (Floating):"
                file.write(("// Population " + index + ' Synaptic Params' + '\n'))
                print("\nENSEMBLE", index, ": SYNAPSES\n")
                # Weight Params
                file.write(('parameter ' + 'N_WEIGHT_' + index + ' = ' + str(sec.n_w) + ',' + '\n'))
                print(report_tag, "Weight bit depth: ", sec.n_w)
                file.write(('SCALE_W_' + index + ' = ' + str(sec.scale_w) + ',' + '\n'))
                print(report_tag, "Weight scale value: ", sec.scale_w)
                file.write(('N_WEIGHT_EXP_' + index + ' = ' + str(sec.n_w_exp) + ',' + '\n'))
                print(report_tag, "Weight exponent depth: ", sec.n_w_exp)
                # Activation Params
                file.write(('N_ACTIV_EXTRA_' + index + ' = ' + str(sec.n_activ_extra) + ',' + '\n'))
                print(report_tag, "Extra bit depth in weight path: ", sec.n_activ_extra)
                # Synaptic time constant
                file.write(('PSTC_SHIFT_' + index + ' = ' + str(sec.pstc_shift) + ';' + '\n'))
                print(report_tag, "Post-synaptic time constant shift: ", sec.pstc_shift)
                # file.write(('ACTIV_L_SHIFT_' + index + ' = ' + str(sec.activ_l_shift) + ';' + '\n')) Obsolete
                file.write('\n')
                pop_count += 1
            elif class_name == 'Encoder_Floating':
                index = pop_names[pop_count]
                report_tag += index + " Encoder:"
                print("\nENSEMBLE", index, ": FLOATING-POINT ENCODER NEURONS\n")
                file.write(("// Population " + index + ' Params' + '\n'))
                file.write(('parameter ' + 'N_NEURON_' + index + ' = ' + str(sec.n_neurons) + ',' + '\n'))
                print(report_tag, "Number of neurons: ", sec.n_neurons)
                # X/Incoming activation Params
                file.write(('N_X_' + index + ' = ' + str(sec.n_x) + ',' + '\n'))
                print(report_tag, "Input bit depth: ", sec.n_x)
                file.write(('RADIX_X_' + index + ' = ' + str(sec.radix_x) + ',' + '\n'))
                print(report_tag, "Input radix: ", sec.radix_x)
                # Gain Params
                file.write(('N_G_MAN_' + index + ' = ' + str(sec.n_g_mantissa) + ',' + '\n'))
                print(report_tag, "Gain mantissa bit depth: ", sec.n_g_mantissa)
                file.write(('N_G_EXP_' + index + ' = ' + str(sec.n_g_exponent) + ',' + '\n'))
                print(report_tag, "Gain exponent bit depth: ", sec.n_g_exponent)
                # Bias Params
                file.write(('N_B_MAN_' + index + ' = ' + str(sec.n_b_mantissa) + ',' + '\n'))
                print(report_tag, "Bias mantissa bit depth: ", sec.n_b_mantissa)
                file.write(('N_B_EXP_' + index + ' = ' + str(sec.n_b_exponent) + ',' + '\n'))
                print(report_tag, "Bias exponent bit depth: ", sec.n_b_exponent)
                # Output value to NAU param
                file.write(('N_DV_POST_' + index + ' = ' + str(sec.n_dv_post) + ';' + '\n'))
                print(report_tag, "Bit depth of datapath to the NAU: ", sec.n_dv_post)
                file.write('\n')

        print("\nGLOBAL PARAMETERS: \n")
        file.write(("// Global Synaptic Params" + '\n'))
        file.write(('parameter N_R = ' + str(global_params[0])  + ',' + '\n'))
        print("Bit depth of refractory period: ", global_params[0])
        file.write(('REF_VALUE = ' + str(global_params[1])  + ',' + '\n'))
        print("Corresponding value of refractory period: ", global_params[1])
        file.write(('T_RC_SHIFT = ' + str(global_params[2])  + ';' + '\n'))
        print("Synaptic RC filter time constant shift value: ", global_params[2])

        print("\n/-----------------------------/\n")