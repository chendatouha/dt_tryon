import argparse, os, sys, datetime
from omegaconf import OmegaConf
import pprint


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        default='config.yaml'
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    return parser

def shared_section(cfg, logdir):
    from torch.utils.data import DataLoader
    dataset_train = instantiate_from_config(cfg.data.trainset)
    data_loader_train = DataLoader(dataset_train, batch_size=cfg.data.trainset.batch_size, shuffle=True,
                                   num_workers=cfg.data.trainset.workers, drop_last=True)

    dataset_val = instantiate_from_config(cfg.data.valset)
    data_loader_val = DataLoader(dataset_val, batch_size=cfg.data.valset.batch_size, shuffle=False,
                                 num_workers=cfg.data.valset.workers, drop_last=True)
    trainer = instantiate_from_config(cfg.trainer)
    trainer.train(data_loader_train, data_loader_val, cfg.train_cfg, logdir)
    print('finish')











if __name__ == "__main__":
    import shutil
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.cfg)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    nowname = now + '_' + cfg_name
    logdir = os.path.join(opt.logdir, nowname)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir,'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir,'val'), exist_ok=True)

    OmegaConf.register_new_resolver('logdir', lambda x: os.path.join(logdir, x))
    config = OmegaConf.load(opt.cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.devices
    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        os.makedirs(os.path.join(logdir, 'config'), exist_ok=True)
        shutil.copy(opt.cfg, os.path.join(logdir, 'config', cfg_fname))
        print('\n*****config*****')
        pprint.pprint(config, compact=True)
    from utils.train_utils import instantiate_from_config, set_seeds
    set_seeds(opt.seed)
    shared_section(config, logdir)