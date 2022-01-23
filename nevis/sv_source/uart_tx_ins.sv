always @(posedge clk) begin
        uart_tx_data[<bit_post> : <bit_pre>] <= o_d_valid_conn_<i_pre>C<i_post> ? o_d_conn_<i_pre>C<i_post>[<i_dim>] : uart_tx_data[<bit_post> : <bit_pre>];
        uart_tx_valid[<valid_i>] <= o_d_valid_conn_<i_pre>C<i_post> ? 1'b1 : uart_tx_valid[<valid_i>];
    end

    <tx-flag>