import torch

def get_dataiter(data_name, root, set, net_name, config, max_rul=125, seq_len=30, mod='normal'):

    if mod == 'normal':
        from .data_loader_v2 import CMPDataIter, worker_init_fn
        data_iter = CMPDataIter(root, set, max_rul, seq_len, net_name)
    elif mod == 'clip':
        from .data_loader_prompt import CMPDataIter, worker_init_fn
        data_iter = CMPDataIter(root, set, max_rul, seq_len, net_name)

    data_loader = torch.utils.data.DataLoader(data_iter, batch_size=config.train.batch_size,
                                                 num_workers=config.data.num_worker,
                                                 pin_memory=True, worker_init_fn=worker_init_fn)
    return data_loader, data_iter