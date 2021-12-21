    /*************** DECODER <i_pre>C<i_pre> *****************/
    connection_top #(
        .N_WEIGHT(N_WEIGHT_0C1),
        .SCALE_W(SCALE_W_0C1),
        .N_WEIGHT_EXP(N_WEIGHT_EXP_0C1),
        .N_NEURON_PRE(N_NEURON_PRE_0C1),
        .N_NEURON_POST(1),
        .PSTC_SHIFT(PSTC_SHIFT_0C1),
        .N_ACTIV_EXTRA(N_ACTIV_EXTRA_0C1),
        .OUTPUT_DIMS(OUTPUT_DIMS_0C1),
        .N_OUTPUT(N_OUTPUT_0C1),
        .FILE_ID_PRE(0),
        .FILE_ID_POST(1)
    ) connection_1 (
        .clk(clk),
        .rst(rst),
        .global_pulse(global_pulse_dly),
        
        .i_spike_addr(o_spike_bus_0),
        //.i_addr_valid(o_addr_valid_0),
        .i_fifo_empty(o_fifo_empty_0),
        .o_fifo_rd_en(i_rd_en_0),
        
        .i_prev_idle(o_idle_0),
        .o_activate_encoder(),
        
        .o_output_bus(o_output_val),
        .o_output_valid(o_output_valid),
        .o_idle()
    );