    /************* ENSEMBLE <i> WIRES *************/
    wire [$clog2(N_NEURON_<i>)-1:0] o_spike_bus_<i>;
    wire signed [N_X_<i>-1:0] i_d_ensemble_<i> [INPUT_DIMS_<i>-1:0][INPUT_NUM_<i>-1:0];

