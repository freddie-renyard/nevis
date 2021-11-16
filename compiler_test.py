from proto_nengo.example_source import ModelCreator # Import the 'Nengo - Under the Hood' example code
from nevis.memory_compiler import Compiler # Import the compiler code
from nevis.filetools import Filetools
from nevis.neuron_classes import Encoder, Encoder_Floating, Synapses, Synapses_Floating
from nevis.global_tools import Global_Tools
import os
from subprocess import call

def target_function(x):
    """The function to compute between A and B"""
    return x**3

dt = 0.001

n_value = 16 #16 this is t_rc after scaling by dt
t_rc = Global_Tools.inverse_rc(n_value, dt) # membrane RC time constant

n_value = 128 # Produces a t_pstc of 0.1275 (4sf) 
t_pstc = Global_Tools.inverse_pstc(n_value, dt) # post-synaptic time constant

# Create the neuron model from the Nengo example code
model = ModelCreator(
    dt          = 0.001, 
    t_ref       = 0.002,
    t_pstc      = t_pstc, 
    N_A         = 50, # number of neurons in first population
    N_B         = 40, # number of neurons in second population
    N_samples   = 100, # number of sample points to use when finding decoders
    rate_A      = [25, 75], # range of maximum firing rates for population A
    rate_B      = [50, 100], # range of maximum firing rates for population B
    t_rc        = t_rc,
    target_function=target_function,
    seed        = 43
)

''' Plot the dynamic range of some model data
data = [
    model.gain_A,
    model.decoder_A,
    model.gain_B,
    model.decoder_B,
]

Global_Tools.plot_dynamic_range(data, 100)

model.run_simulation()
'''

# Compile network encoder ensemble to specified parameters
n_x = 8
radix = 6
radix_w = 10
n_dv_post = 10

encoder_A = Encoder_Floating(
    n_neurons       = model.N_A,
    gain_list       = model.gain_A,
    bias_list       = model.bias_A,
    encoder_list    = model.encoder_A,
    t_rc            = model.t_rc,
    n_x             = n_x,
    radix_x         = n_x-1,
    radix_g         = radix,
    radix_b         = radix,
    n_dv_post       = n_dv_post,
    verbose         = False
)

#model.monitor_spikes = True
#model.make_sim_plots()
#exit()

'''
synapses_B = Synapses(
    n_neurons=model.N_B,
    pstc_scale=model.pstc_scale,
    decoders_list=model.decoder_A,
    encoders_list=model.encoder_B,
    n_activ_extra=1,
    radix_w=radix_w,
    verbose=True
)

'''
synapses_B = Synapses_Floating(
    n_neurons       = model.N_B,
    pstc_scale      = model.pstc_scale,
    decoders_list   = model.decoder_A,
    encoders_list   = model.encoder_B,
    n_activ_extra   = 6,
    radix_w         = radix_w,
    minimum_val     = 0,
    verbose         = True
)

# Compute the required bit depth of the input to the second population's encoder
#n_x_b = synapses_B.n_w + synapses_B.n_activ_extra + synapses_B.scale_w + 1 #Fixed point version

n_x_b = synapses_B.n_w_man + synapses_B.n_activ_extra + synapses_B.scale_w + 1

encoder_B = Encoder_Floating(
    n_neurons       = model.N_B,
    gain_list       = model.gain_B,
    bias_list       = model.bias_B,
    encoder_list    = model.encoder_B,
    t_rc            = model.t_rc,
    n_x             = n_x_b,
    radix_x         = n_x_b-2, # minus two to bring the input register range to -2 to 2
    radix_g         = radix,
    radix_b         = radix,
    n_dv_post       = n_dv_post,
    verbose         = False
)

'''
output_stage = Synapses(
    n_neurons=1,
    pstc_scale=model.pstc_scale,
    decoders_list=model.decoder_B,
    encoders_list=[1], # Indicates a positive weight addition
    n_activ_extra=1,
    radix_w=radix_w,
    verbose = True
)
'''

output_stage = Synapses_Floating(
    n_neurons       = 1,
    pstc_scale      = model.pstc_scale,
    decoders_list   = model.decoder_B,
    encoders_list   = [1], # Indicates a positive weight addition
    n_activ_extra   = 8,
    radix_w         = radix_w,
    minimum_val     = 0,
    verbose         = False
)

ref_period, n_r = Compiler.calculate_refractory_params(model.t_ref*model.original_dt, model.original_dt)
t_rc_hardware = Compiler.calculate_t_rc_shift(model.t_rc)

compiled_model = [encoder_A, synapses_B, encoder_B, output_stage]
refractory_params = [n_r, ref_period, t_rc_hardware]

# Save the parameters as a Verilog header file.
Filetools.compile_and_save_header(
    filename = 'model_params.vh', 
    full_model = compiled_model, 
    global_params = refractory_params
)

# Save the compiled parameters in the appropriate files
running_mem_total = encoder_A.save_params(0, n_r=n_r)
running_mem_total = synapses_B.save_params(index=1, running_mem_total=running_mem_total)
running_mem_total = encoder_B.save_params(index=1, n_r=n_r, running_mem_total=running_mem_total)
running_mem_total = output_stage.save_params(index=2, running_mem_total=running_mem_total)

Filetools.report_memory_usage(running_mem_total)

run_vivado = input("Would you like to run Vivado on host machine? [y/n]: ")
if run_vivado == 'y':
    # Call the script that transfers the compiled files to 
    # the Vivado server machine
    cwd = os.getcwd()
    
    script_path = cwd + "/nevis/File_transfer.sh"
    call(script_path)

# Open the serial interface. The board will need to be 
# plugged in to the Vivado machine and programmed from 
# there as the drivers are incompatible with macOS. 
run_model = (input("\nWould you like to run the model alongside the FPGA's output? [y/n]: ") == 'y')
model.monitor_spikes = False

from nevis.serial_interface import run_serial_interface
run_serial_interface(model=model, 
    run_model=run_model, 
    out_depth = (output_stage.n_w_man + output_stage.scale_w + output_stage.n_activ_extra + 1),
    output_scale = (output_stage.n_w_man - 1 + output_stage.scale_w + output_stage.n_activ_extra)
)