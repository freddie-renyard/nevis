module nevis_compiled(
    input clk,
    input rst,
    
    input rx,
    output tx,
    output [7:0] debug_data,

    output [$clog2(N_NEURON_2)-1:0] o_spike_addr,
    output o_spike,
    output debug_d_valid
    );
    
    `include "model_params.vh"
    
    wire global_pulse;
    localparam CLK_FREQ_MHZ = 100;

