import torch

def get_dataiter(data_name, config, mod='normal'):

    root = config.data.root
    set = config.data.set
    net_name = config.net.name
    max_rul = config.data.max_rul
    seq_len = config.data.seq_len
    n_user = config.fed.n_user
    sample_interval = config.fed.sample_interval
    
    
    if mod == 'normal':
        from .data_loader_v2 import CMPDataIter, worker_init_fn
        data_iter = CMPDataIter(root, set, max_rul, seq_len, net_name)
    elif mod == 'clip':
        from .data_loader_prompt import CMPDataIter, worker_init_fn
        data_iter = CMPDataIter(root, set, max_rul, seq_len, net_name)
    elif mod == 'fed':
        from .data_loader_fed_iid import CMPDataIterFed, worker_init_fn
        data_iter = CMPDataIterFed(root, set, max_rul, seq_len, net_name, n_user, sample_interval)

    data_loader = torch.utils.data.DataLoader(data_iter, batch_size=config.train.batch_size,
                                                 num_workers=config.data.num_worker,
                                                 pin_memory=True, worker_init_fn=worker_init_fn)
    return data_loader, data_iter