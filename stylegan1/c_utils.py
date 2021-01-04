
import torch



save_device = 'cpu'
load_device = 'cpu'


def set_device(device):
    
    global save_device, load_device
    save_device = device
    load_device = device
    if device=='gpu':
        device = 'cuda:0'
    return torch.device(device)


def print_device() : print(save_device, load_device)


def apply_wn(module, wn=None) : return module if wn==None else wn(module)


def wrap_weight_norm(weight_norm_, *args, **kwargs):
    return lambda module: weight_norm_(module, *args, **kwargs)


def calc_pool2d_pad(size, kernel_size, stride):

    h = size
    w = size
    if (isinstance(size, list) or isinstance(size, tuple)):
        h = size[0]
        w = size[1]
    h_pad = (h // stride - 1)*stride + (h % stride) - h + kernel_size
    w_pad = (w // stride - 1)*stride + (h % stride) - h + kernel_size

    l_pad = h_pad // 2
    r_pad = h_pad // 2 if (h_pad % 2) == 0 else h_pad // 2 + 1
    t_pad = w_pad // 2
    b_pad = w_pad // 2 if (w_pad % 2) == 0 else w_pad // 2 + 1
    return (l_pad, r_pad, t_pad, b_pad)

def load_state_dict_to_model(model, s_dict, save_device, load_device):
    if save_device != load_device:
        model.load_state_dict(s_dict, map_location=load_device)
    else:
        model.load_state_dict(s_dict)
    if load_device == torch.device('cuda:0'):
        model.to(load_device)


def load_model(model_class, model_dict, path=None):

    s_dict = model_dict
    if path!=None:
        s_dict = torch.load(path)
    arguments = s_dict["arguments"]
    model = model_class(**arguments)

    if save_device != load_device:
        model.load_state_dict(s_dict, map_location=torch.device('cpu') if load_device=='cpu' else "cuda:0")
    else:
        model.load_state_dict(s_dict)
    if load_device=='gpu':
        model.to(torch.device('cuda'))
    '''
    if mode=='eval':
        return model.eval()
    elif mode=='train':
        return model.train()
    else:
        return model
    '''
    return model