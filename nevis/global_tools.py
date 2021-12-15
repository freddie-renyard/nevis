import math
from matplotlib import pyplot as plt
import numpy as np

class Global_Tools:

    """
    This class contains functions for determining optimal values for certain
    Nengo netowrk parameters to ensure optimal hardware performance. Unlike 
    the rest of the compilation process, these must be run before the model is
    created in Nengo.
    """

    def inverse_pstc(n_value, dt):
        '''
        This function creates the corrent t_pstc value from a specified power of 2. This allows for
        vast reduction in hardware complexity by using arithmetic right-shifts.
        '''
        inv_result = math.log(1 - 1/n_value)
        return -dt/inv_result

    def inverse_rc(n_value, dt):
        '''
        This function creates the corrent t_rc value from a specified power of 2. This allows for
        vast reduction in hardware complexity by using arithmetic right-shifts.    
        '''
        inv_result = 1/n_value
        return dt/inv_result

    def plot_dynamic_range(input_list, bins, name="[no name given]"):
        
        data_no = len(input_list)

        i = 1
        for data_set in input_list:
            plt.subplot(1, data_no, i)
            plt.hist(data_set, bins=bins, histtype=u"step")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            #plt.title("Dynamic range of parameter " + name)
            i += 1
        
        plt.show()