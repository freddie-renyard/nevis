module nevis_top(
    input clk,
    input rst,
    
    input rx,
    output tx,
    
    // Debug signals
    output o_spike,
    output [$clog2(N_NEURON_0)-1:0] o_spike_addr,
    output debug_d_valid
    );
    
    `include "model_params.vh"
    
    wire global_pulse;
    localparam BAUD_RATE = 2000000;
    localparam CLK_FREQ_MHZ = 100;

