    /************* UART ***********/
    
    // Deconcatenate the data to be transmitted. This is done to allow easier interfacing
    // with multiple ensembles later on in development.
    localparam N_DATA_TX    = N_CONN_OUT * TX_NUM_OUTS;   // Compile this
    
    wire        [N_DATA_TX-1:0] uart_tx_data;
    wire signed [N_X_0*INPUT_DIMS_0-1:0] uart_rx_data;
    
    <tx-flag>

    uart_top #(
        .BAUD_RATE(BAUD_RATE),
        .CLK_FREQ_MHZ(CLK_FREQ_MHZ),
        .TX_N_DATA_WORD(N_TX),
        .TX_WORDS(TX_NUM_OUTS),
        .RX_N_DATA_WORD(N_X_0),
        .RX_WORDS(INPUT_DIMS_0)
    ) uart ( 
        .clk(clk),
        .rst(rst),
        .rx(rx),
        .tx(tx),
        .i_data(uart_tx_data),
        .i_new_data(o_output_valid),
        .o_data(uart_rx_data),
        .o_new_data(rx_new_data),
        .i_block(1'b0),
        .o_busy(o_uart_busy)
    );
    
     // Reformat the rx data to the ensemble shape.
    genvar j;
    generate
        for (j = 0; j < INPUT_NUM_0; j = j + 1) begin
            for (i = 0; i < INPUT_DIMS_0; i = i + 1) begin
                assign i_d_ensemble_0[i][j] = uart_rx_data[i*N_X_0 +: N_X_0];
            end
        end
    endgenerate