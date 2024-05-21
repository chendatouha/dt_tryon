import time
import importlib


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def time_it(fun, *args):
    time_start = time.time()
    result = fun(*args)
    time_used = time.time() - time_start
    return result, time_used


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def log_captions(captions, file_name):
    with open(file_name, 'w', encoding='utf8') as fp:
        for idx, c in enumerate(captions):
            fp.write('{}: {}\n\n'.format(idx, c))

if __name__ == '__main__':
    pass