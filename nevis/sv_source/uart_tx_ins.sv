assign uart_tx_data[<bit_post> : <bit_pre>] = o_d_conn_<i_pre>C<i_post>[<i_dim>];
    assign uart_tx_valid[<valid_i>] = o_d_valid_conn_<i_pre>C<i_post>[0];

    <tx-flag>