import random
import numpy as np
import torch
import importlib
import accelerate


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerate.utils.set_seed(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device, n_gpu


def change_option(cfg, name, value):
    cfg[name] = value
    print('change {} to {}'.format(name, value))


def Print(msg, rank):
    if rank in [-1, 0]:
        print(msg)
    else:
        pass


def init_from_ckpt(model, path, ignore_keys=None, state_dict_key=None):
    if ignore_keys is None:
        ignore_keys = []
    sd = torch.load(path, map_location="cpu")()
    if state_dict_key is not None:
        sd = sd[state_dict_key]
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys:\n {missing}")
    if len(unexpected) > 0:
        print(f"\nUnexpected Keys:\n {unexpected}")

    return model


def log_GPU_memory_usage(local_rank):
    mem_loc = torch.cuda.memory_allocated() // 1024 ** 3
    max_mem_loc = torch.cuda.max_memory_allocated() // 1024 ** 3
    mem_reserve = torch.cuda.memory_reserved() // 1024 ** 3
    max_mem_reserve = torch.cuda.max_memory_reserved() // 1024 ** 3
    print("GPU mem statistic@rank{}: loc:{}G max_loc:{}G reserve:{}G max_reserve{}G".format(local_rank,mem_loc,
                                                                                            max_mem_loc, mem_reserve, max_mem_reserve))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if not "params" in config:
        raise KeyError('Expected key `params` to instantiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_trainable_parameters(model):
    trainable_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module