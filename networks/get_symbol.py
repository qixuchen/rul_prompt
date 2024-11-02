
def get_symbol(net_name, config, num_hidden=18, input_dim=14, aux_dim=4, hand_craft=False, pmpt_1=None):
    if net_name == 'cnn_b':
        from .cnn_b6 import CNN_B
        sym_net = CNN_B(num_hidden=num_hidden, input_dim=input_dim, aux_dim=aux_dim)
    elif net_name == 'vit':
        from .trans_1 import TST
        sym_net = TST(in_length=config.data.seq_len, in_dim=input_dim, embed_dim=16, dim=128,
                      depth=12, heads=12, mlp_dim=256, pool='reg', dim_head=64, dropout=0.2, emb_dropout=0.)
    elif net_name == 'tst':
        from .trans_1 import TST
        sym_net = TST(in_length=config.data.seq_len, in_dim=input_dim, embed_dim=16, dim=128,
                      depth=6, heads=6, mlp_dim=256, pool='reg', dim_head=64, dropout=0.2, emb_dropout=0.)
    elif 'clip' in net_name:
        if net_name == 'bilstm_clip':
            from .bi_lstm_clip import Bi_LSTM_CLIP
            sym_net = Bi_LSTM_CLIP(input_dim=config.net.input_dim, prompt_dict=pmpt_1)
        elif net_name == 'tst_clip':
            from networks.trans_clip import TST_CLIP
            sym_net = TST_CLIP(in_length=config.data.seq_len, in_dim=config.net.input_dim, embed_dim=16, dim=128,
                               depth=6, heads=6, mlp_dim=256, prompt_dict=pmpt_1, pool='reg', dim_head=64,
                               dropout=0.2, emb_dropout=0.)
    else:
        from .bi_lstm_org import Bi_LSTM
        sym_net = Bi_LSTM(num_hidden=num_hidden, input_dim=input_dim, aux_dim=aux_dim)

    return sym_net