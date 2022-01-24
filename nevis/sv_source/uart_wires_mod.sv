    /************* UART ***********/
    
    // Deconcatenate the data to be transmitted. This is done to allow easier interfacing
    // with multiple ensembles later on in development.
    localparam N_DATA_TX    = N_TX * TX_NUM_OUTS; 
    localparam N_DATA_RX    = N_RX * RX_NUM_INS; 
    
    reg  signed [N_DATA_TX-1:0]     uart_tx_data;
    reg         [TX_NUM_OUTS-1:0]   uart_tx_valid;

    wire signed [N_DATA_RX-1:0] uart_rx_data;
    
    <tx-flag>

    // Reset the transmit register if reset is asserted or all the data is valid.
    /*
    always @(posedge clk) begin
        if (rst | &uart_tx_valid) begin
            uart_tx_valid <= 0;
        end
    end
    */

    // Output the data if all of the bits in the validity register are high.
    wire output_valid;
    assign output_valid = &uart_tx_valid;

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
        .i_new_data(output_valid),
        .o_data(uart_rx_data),
        .o_new_data(rx_new_data),
        .i_block(1'b0),
        .o_busy(o_uart_busy)
    );

    basic_scheduler scheduler (
        .clk(clk),
        .rst(rst),
        .i_block(o_uart_busy),
        .i_trigger(rx_new_data),
        .o_global_pulse(o_scheduler)
    );

    assign debug_data = o_d_conn_2C1[0];

    <rx-flag>
