import torch

def select_optimizer(optim: str, params, lr: float):
    optim = optim.lower()
    if optim == 'sgd':
        optimizer = torch.optim.SGD(params, 
									lr=lr, 
									momentum=0.9, 
									weight_decay=0.0001, 
									nesterov=True)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=lr,
                                    weight_decay=0.0001)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                    lr=lr,
                                    weight_decay=0.0001)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params,
                                    lr=lr,
                                    weight_decay=0.0001)
    else:
        raise NotImplementedError(optim)
    
    return optimizer