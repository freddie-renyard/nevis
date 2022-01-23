    /************* ENSEMBLE <i> MODULE ************/
    ensemble_top #(
            .N_X(N_X_<i>),
            .RADIX_X(RADIX_X_<i>),
            .N_PHI(N_PHI_<i>),
            .N_G_MAN(N_G_MAN_<i>),
            .N_G_EXP(N_G_EXP_<i>),
            .N_B_MAN(N_B_MAN_<i>),
            .N_B_EXP(N_B_EXP_<i>),
            
            .INPUT_DIMS(INPUT_DIMS_<i>),
            .INPUT_NUM(INPUT_NUM_<i>),
            
            .N_DV_POST(N_DV_POST_<i>),
            .N_R(N_R_<i>), 
            .N_NEURON(N_NEURON_<i>),
            .T_RC_SHIFT(T_RC_SHIFT_<i>),
            .REF_VALUE(REF_VALUE_<i>),
            
            .FILE_ID(<i>)
        ) ensemble_0 (
            .clk(clk),
            .rst(rst),
            //.global_pulse(<pulse>),
            
            .i_x(i_d_ensemble_<i>),
            .o_idle(o_idle_<i>),
            .i_d_valids(i_d_valids_<i>),
            
            .o_fifo_empty(o_fifo_empty_<i>),
            .o_data_bus(o_spike_bus_<i>),
            .i_rd_en(i_rd_en_<i>),
            .o_global_pulse_dly(global_pulse_dly_<i>)
        );

