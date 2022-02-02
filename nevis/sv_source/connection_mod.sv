    /*************** DECODER <i_pre>C<i_post> *****************/
    connection_top #(
        .N_WEIGHT(N_WEIGHT_<i_pre>C<i_post>),
        .SCALE_W(SCALE_W_<i_pre>C<i_post>),
        .N_WEIGHT_EXP(N_WEIGHT_EXP_<i_pre>C<i_post>),
        .N_NEURON_PRE(N_NEURON_PRE_<i_pre>C<i_post>),
        .N_NEURON_POST(1),
        .PSTC_SHIFT(PSTC_SHIFT_<i_pre>C<i_post>),
        .RECURRENT(<recurrent>),
        .RADIX_OUTPUT(RADIX_OUTPUT_<i_pre>C<i_post>),
        .N_ACTIV_EXTRA(N_ACTIV_EXTRA_<i_pre>C<i_post>),
        .OUTPUT_DIMS(OUTPUT_DIMS_<i_pre>C<i_post>),
        .N_OUTPUT(N_OUTPUT_<i_pre>C<i_post>),
        .FILE_ID_PRE(<i_pre>),
        .FILE_ID_POST(<i_post>)
    ) connection_<i_pre>C<i_post> (
        .clk(clk),
        .rst(rst),
        .global_pulse(global_pulse_dly_<i_pre>),
        
        .i_spike_addr(o_spike_bus_<i_pre>),
        //.i_addr_valid(o_addr_valid_0),
        .i_fifo_empty(o_fifo_empty_<i_pre>),
        .o_fifo_rd_en(<read_enable>/*i_rd_en_<i_pre>*/),
        
        .i_prev_idle(o_idle_<i_pre>),
        .o_activate_encoder(activate_encoder_<i_pre>C<i_post>),
        
        .o_output_bus(o_d_conn_<i_pre>C<i_post>),
        .o_output_valid(o_d_valid_conn_<i_pre>C<i_post>),
        .o_idle()
    );

