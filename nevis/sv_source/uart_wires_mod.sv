    /************* UART ***********/
    
    // Deconcatenate the data to be transmitted. This is done to allow easier interfacing
    // with multiple ensembles later on in development.
    localparam N_DATA_TX    = N_TX * TX_NUM_OUTS; 
    
    wire        [N_DATA_TX-1:0] uart_tx_data;
    wire signed [N_X_0*INPUT_DIMS_0-1:0] uart_rx_data;
    
    <tx-flag>

    uart_top #(
        .BAUD_RATE(BAUD_RATE),
        .CLK_FREQ_MHZ(CLK_FREQ_MHZ),
        .TX_N_DATA_WORD(N_TX),
        .TX_WORDS(TX_NUM_OUTS),
        .RX_N_DATA_WORD(N_RX),
        .RX_WORDS(RX_NUM_INS)
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

    <rx-flag>
