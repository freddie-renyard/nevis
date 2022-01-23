module nevis_top(
    input clk,
    input rst,
    
    input rx,
    output tx
    );
    
    `include "model_params.vh"
    
    wire global_pulse;
    localparam CLK_FREQ_MHZ = 100;

