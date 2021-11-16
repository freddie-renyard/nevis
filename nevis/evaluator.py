from matplotlib import pyplot as plt
import numpy as np

class Evaluator:
    """
    This class contains methods to assist with the evaluation of the hardware outputs against
    the software output.
    """

    def decode_hardware_output(self, filename, radix, verbose=False):
        """Takes the file specified by filename and turns its data into a list
        for direct comparision against simulator output. The file is from the total
        model SystemVerilog testbench, and consists of the unscaled integer output
        of the module delimited with line breaks.

        Parameters
        ----------
        filename: str
            The filename of the target hardware evaluation data.
        radix: int
            The place of the radix point for the hardware's fixed point 
            integer output register.
        verbose: bool
            A flag that can be set to show each value against it's scaled
            counterpart and index in the file.
        
        """
        output_list = []
        file = open(filename)

        i = 0
        for line in file:
            integer_val = int(line)
            float_val = float(integer_val) / (2 ** radix)
            output_list.append(float_val)
            
            if verbose:
                print("Index: ", i, " ", integer_val, "-->", float_val)

            i += 1
        
        return output_list

    def error_comparison(self, list_0, list_1, period=0):
        """ Computes the differences of each pair of values in the lists at
        the same list indices. 

        Parameters
        ----------
        list_0: [float or int]
            The list of predicted (software model) values.
        list_1: [float or int]
            The list of observed (hardware model) values.
        period: int
            The number of steps to compute the MSE over. 
            If set to 0, the difference between each timestep is computed.
            Default: 0.
        
        Returns
        -------
        error_list: [float]
            A list of the errors computed at all the indices
            of the shortest list.
        """
        if len(list_0) != len(list_1):
            print("Evaluator: List lengths are unequal. Evaluating from both lists' index 0...")

        error_list = []
        if period == 0:
            for elements in zip(list_0, list_1):
                error = elements[1] - elements[0]
                error_list.append(error)
        else:
            i = 0
            mse_value = 0
            for elements in zip(list_0, list_1):
                error = elements[1] - elements[0]
                error = error ** 2
                mse_value += error

                if i % period == 0:
                    mse_value /= period
                    error_list.append(mse_value)
                    mse_value = 0
                i += 1
                
        return error_list
        
    def plot_error_graph(self, error_list):
        """Plots a graph of each error per timestep.
        
        Parameters
        ----------
        error_list: [float]
            The list of errors to be plotted.         
        """

        x = np.linspace(0.2, 10, 100)
        fig, ax = plt.subplots()
        ax.plot(range(len(error_list)), error_list)
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_xlabel("Timestep (dt)")
        ax.set_ylabel("Difference between expected and observed values")

        plt.show()
