import math
import random
from tkinter.constants import E

import numpy
import matplotlib.pyplot as plt

def input(t):
    """The input to the system over time"""
    return 1.0 # math.sin(t * 0.001)

class ModelCreator:

    """
    This is the code for the 'Under the Hood' example given at:
    (https://www.nengo.ai/nengo/examples/advanced/nef-algorithm.html)
    The format of the code has been changed into class object format
    rather than the original script to allow for the compiler to use
    model parameters as attributes for more transferable system design.

    Other aspects of the algorithm of the code have been changed to reflect
    the functioning of the model in hardware. These are marked with the
    keyword EDIT for comparision.
    """

    def __init__(self, dt = 0.001, t_rc = 0.02, t_ref = 0.002, t_pstc = 0.1, 
        N_A = 50, N_B = 40, N_samples = 100, rate_A = [25, 75], rate_B = [50, 100], sim_test_function = lambda x, t: math.sin(x * t), target_function = lambda x: x, seed=11):

        # Set seed to allow for easy evaluation of hardware without stochastic parameters
        # changing between compiler runs.
        random.seed(seed) # was 11
        print("Creating model...")
        '''
        EDIT: all the module parameters are normalised with the timestep, to allow
        the timestep to be set to 1 which is how the hardware functions.
        '''
        self.t_rc = t_rc / dt # membrane RC time constant
        self.t_ref = t_ref / dt # refractory period
        self.t_pstc = t_pstc / dt # post-synaptic time constant
        self.N_A = N_A # number of neurons in first population
        self.N_B = N_B  # number of neurons in second population
        self.N_samples = N_samples  # number of sample points to use when finding decoders
        self.rate_A = [x for x in rate_A] # range of maximum firing rates for population A
        self.rate_B = [y for y in rate_B]  # range of maximum firing rates for population B
        self.target_function = target_function
        self.sim_test_fx = sim_test_function

        # Parameter for monitoring the simulation output for hardware debugging
        self.monitor_spikes = False

        # Having calculated all the normalised values for hardware,
        # set the timestep to 1.
        self.original_dt = dt
        self.dt = dt / dt 

        # Simulation vars. These are stored here to allow for 1 timestep progression
        # of the simulation.
        self.v_A = [0.0] * self.N_A  # voltage for population A
        self.ref_A = [0.0] * self.N_A  # refractory period for population A
        
        self.v_B = [0.0] * self.N_B  # voltage for population B
        self.ref_B = [0.0] * self.N_B  # refractory period for population B

        self.input_A = [0.0] * self.N_A  # input for population A
        self.input_B = [0.0] * self.N_B  # input for population B

        self.output = 0.0 # the decoded output value from population B

        # scaling factor for the post-synaptic filter
        self.pstc_scale = 1.0 - math.exp(-self.dt / self.t_pstc)

        # create random encoders for the two populations
        self.encoder_A = [random.choice([-1, 1]) for i in range(N_A)]
        self.encoder_B = [random.choice([-1, 1]) for i in range(N_B)]

        # random gain and bias for the two populations
        self.gain_A, self.bias_A = self.generate_gain_and_bias(N_A, -1, 1, self.rate_A[0], self.rate_A[1])
        self.gain_B, self.bias_B = self.generate_gain_and_bias(N_B, -1, 1, self.rate_B[0], self.rate_B[1])

        # find the decoders for A and B
        self.decoder_A = self.compute_decoder(self.encoder_A, self.gain_A, self.bias_A, function=self.target_function)
        self.decoder_B = self.compute_decoder(self.encoder_B, self.gain_B, self.bias_B)

        # compute the weight matrix
        self.weights = numpy.dot(self.decoder_A, [self.encoder_B])

    def generate_gain_and_bias(self, count, intercept_low, intercept_high, rate_low, rate_high):
        print("Generating gain and bias...")
        gain = []
        bias = []
        for _ in range(count):
            # desired intercept (x value for which the neuron starts firing
            intercept = random.uniform(intercept_low, intercept_high)
            # desired maximum rate (firing rate when x is maximum)
            rate = random.uniform(rate_low, rate_high)

            # this algorithm is specific to LIF neurons, but should
            # generate gain and bias values to produce the desired
            # intercept and rate
            z = 1.0 / (1 - math.exp((self.t_ref - (1.0 / rate)) / self.t_rc))
            g = (1 - z) / (intercept - 1.0)
            b = 1 - g * intercept
            gain.append(g)
            bias.append(b)
        return gain, bias

    def run_neurons(self, input, v, ref, simulating=False):
        """Run the neuron model.

        A simple leaky integrate-and-fire model, scaled so that v=0 is resting
        voltage and v=1 is the firing threshold.
        """
        spikes = []
        for i, _ in enumerate(v):

            if simulating == True:
                dV = self.dt * (input[i] - v[i] / self.t_rc)
            else:
                dV =  self.dt * (input[i] - v[i]) / self.t_rc
            
            v[i] += dV
            if v[i] < 0:
                v[i] = 0  # don't allow voltage to go below 0

            if ref[i] > 0:  # if we are in our refractory period
                v[i] = 0  # keep voltage at zero and
                ref[i] -= self.dt  # decrease the refractory period

            if v[i] > 1:  # if we have hit threshold
                spikes.append(True)  # spike
                v[i] = 0  # reset the voltage
                ref[i] = self.t_ref  # and set the refractory period
            else:
                spikes.append(False)
        return spikes

    def compute_response(self, x, encoder, gain, bias, time_limit=500):
        """Measure the spike rate of a population for a given value x."""
        N = len(encoder)  # number of neurons
        v = [0] * N  # voltage
        ref = [0] * N  # refractory period

        # compute input corresponding to x
        input = []
        for i in range(N):
            input.append(x * encoder[i] * gain[i] + bias[i])
            v[i] = random.uniform(0, 1)  # randomize the initial voltage level

        count = [0] * N  # spike count for each neuron

        # feed the input into the population for a given amount of time
        t = 0
        while t < time_limit:
            spikes = self.run_neurons(input, v, ref, simulating=False)
            for i, s in enumerate(spikes):
                if s:
                    count[i] += 1
            t += self.dt
        return [c / time_limit for c in count]  # return the spike rate (in Hz)

    def compute_tuning_curves(self, encoder, gain, bias):
        """Compute the tuning curves for a population"""
        # generate a set of x values to sample at
        x_values = [i * 2.0 / self.N_samples - 1.0 for i in range(self.N_samples)]

        # build up a matrix of neural responses to each input (i.e. tuning curves)
        A = []
        for x in x_values:
            response = self.compute_response(x, encoder, gain, bias)
            A.append(response)
        return x_values, A

    def compute_decoder(self, encoder, gain, bias, function=lambda x: x):
        print("Computing decoders...")
        # get the tuning curves
        x_values, A = self.compute_tuning_curves(encoder, gain, bias)

        # get the desired decoded value for each sample point
        value = numpy.array([[function(x)] for x in x_values])

        # find the optimal linear decoder
        A = numpy.array(A).T
        Gamma = numpy.dot(A, A.T)
        Upsilon = numpy.dot(A, value)
        Ginv = numpy.linalg.pinv(Gamma)
        decoder = numpy.dot(Ginv, Upsilon) / self.dt
        return decoder

    def run_timestep(self, x, timestep, verbose=0, truncate=False, simulating=True):
        """
        EDIT: The code for running individual timesteps has been moved to a 
        seperate method; this is to allow the serial interface to the FPGA
        to plot the simulation results in realtime along with the output
        of the neuromorphic hardware, to allow for direct comparison. This
        requires control over timestep progression. To facilitate this change, 
        some simulation params have been moved to be attirbutes of the parent
        class.
        """

        trunc = 128.0

        for i in range(self.N_A):
            self.input_A[i] = x * self.encoder_A[i] * self.gain_A[i] / self.t_rc + self.bias_A[i] / self.t_rc
            #print(i, self.input_A[i])
            if truncate == True:
                self.input_A[i] = int(self.input_A[i]*trunc) / trunc

        # run population A and determine which neurons spike
        spikes_A = self.run_neurons(self.input_A, self.v_A, self.ref_A, simulating=simulating)
        
        """
        EDIT: the previous line: input_B[j] *= 1.0 - self.pstc_scale
        has been changed. 
        ORIGINAL:
            input = input*(1-scale) is the change per timestep (dt)
            therefore
            y' = y * (1 - scale)
        NEW:
            input = input - input*scale is the change per timestep (dt)
            therefore
            y' = y - y * scale = y * (1 - scale)
        
        The new format reflects the way the hardware computes this step:
        division via a right shift followed by subtraction.
        """
        for j in range(self.N_B):
            self.input_B[j] -= self.input_B[j] * self.pstc_scale
        
        # for each neuron that spikes, increase the input current
        # of all the neurons it is connected to by the synaptic
        # connection weight
        for i, s in enumerate(spikes_A):
            if s:
                for j in range(self.N_B):
                    self.input_B[j] += self.weights[i][j] * self.pstc_scale

        # compute the total input into each neuron in population B
        total_B = [0] * self.N_B
        for j in range(self.N_B):
            total_B[j] = self.gain_B[j] * self.input_B[j] / self.t_rc + self.bias_B[j] / self.t_rc
            if truncate == True:
                total_B[j] = int(total_B[j]*trunc) / trunc
       
        # run population B and determine which neurons`z spike
        spikes_B = self.run_neurons(total_B, self.v_B, self.ref_B, simulating=simulating)

        """
        EDIT: see above.
        """
        self.output -= self.output * self.pstc_scale

        for j, s in enumerate(spikes_B):
            if s:
                self.output += self.decoder_B[j][0] * self.pstc_scale 

        if self.monitor_spikes:
            spike_indices_A = [i for i, x in enumerate(spikes_A) if x == True]
            spike_indices_B = [i for i, x in enumerate(spikes_B) if x == True]
            #if 49 in spike_indices_A:
            print('Timestep:', int(timestep), "Indices of spiking neurons in pop A: ", spike_indices_A)
            print('Timestep:', int(timestep), "Indices of spiking neurons in pop B: ", spike_indices_B)
            
            print("Output:", self.output)
            print("")

        spikes_per_t_A = []
        spikes_per_t_B = []

        spikes_per_t_A.append(numpy.count_nonzero(spikes_A))
        spikes_per_t_B.append(numpy.count_nonzero(spikes_B))

        return x, spikes_A, spikes_B
        
    def run_simulation(self, timesteps=10000, verbose=0, truncate=False):

        # for storing simulation data to plot afterward
        inputs = []
        times = []
        outputs = []
        ideal = []

        spikes_per_t_A = []
        spikes_per_t_B = []

        t = 0
        create_logs = True

        while t < timesteps:  # noqa: C901 (tell static checker to ignore complexity)
            
            # call the input function to determine the input value
            x = input(t)
            x, spikes_A, spikes_B = self.run_timestep(x, t, verbose=verbose, truncate=truncate)
            
            times.append(t)
            inputs.append(x)
            outputs.append(self.output)
            ideal.append(self.target_function(x))
            spikes_per_t_A.append(numpy.count_nonzero(spikes_A))
            spikes_per_t_B.append(numpy.count_nonzero(spikes_B))

            if create_logs == True:
                path_to_temp = "proto_nevis/proto_nengo/temp_eval/"
                if t == 0:
                    file_spikes_a = open(path_to_temp + "spikes_A.txt", 'w+')
                    file_spikes_a.truncate(0)

                    file_spikes_b = open(path_to_temp + "spikes_B.txt", 'w+')
                    file_spikes_b.truncate(0)

                    file_output_a = open(path_to_temp + "output_A.txt", 'w+')
                    file_output_a.truncate(0)

                    file_output_b = open(path_to_temp + "output_B.txt", 'w+')
                    file_output_b.truncate(0)
                else:
                    file_spikes_a = open(path_to_temp + "spikes_A.txt", 'a')
                    file_spikes_b = open(path_to_temp + "spikes_B.txt", 'a')
                    file_output_a = open(path_to_temp + "output_A.txt", 'a')
                    file_output_b = open(path_to_temp + "output_B.txt", 'a')

                for element in [i for i, x in enumerate(spikes_A) if x == True]:
                    file_spikes_a.write(str(element) + " ")
                
                for element in [i for i, x in enumerate(spikes_B) if x == True]:
                    file_spikes_b.write(str(element) + " ")
                
                file_output_a.write(str(self.input_B[0]) + "\n")
                file_output_b.write(str(self.output) + "\n")
                
                file_spikes_a.write('\n')
                file_spikes_b.write('\n')

                file_spikes_a.close()
                file_spikes_b.close()
                file_output_a.close()
                file_output_b.close()

            t += self.dt

        return inputs, times, ideal, outputs, spikes_per_t_A, spikes_per_t_B

    def make_sim_plots(self):
        
        x, A = self.compute_tuning_curves(self.encoder_A, self.gain_A, self.bias_A)
        x, B = self.compute_tuning_curves(self.encoder_B, self.gain_B, self.bias_B)
        inputs, times, ideal, outputs, spikes_a, spikes_b = self.run_simulation(timesteps=5000, verbose=0, truncate=False)

        # Create plots of simulation
        plt.figure()
        plt.plot(x, A)
        plt.title("Tuning curves for population A")

        plt.figure()
        plt.plot(x, B)
        plt.title("Tuning curves for population B")

        plt.figure()
        bins = numpy.linspace(0, 20, 20)
        plt.hist(spikes_a, bins, alpha=0.5, label='Population A')
        plt.hist(spikes_b, bins, alpha=0.5, label='Population B')
        plt.title('Histogram of number of spikes of each population per timestep')

        plt.figure()
        plt.plot(times, inputs, label="input")
        plt.plot(times, ideal, label="ideal")
        plt.plot(times, outputs, label="output")
        plt.title("Simulation results")
        plt.legend()
        plt.show()