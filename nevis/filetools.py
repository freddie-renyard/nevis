import os
import logging

logger = logging.getLogger(__name__)

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