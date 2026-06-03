from swarm_rl.env_wrappers.data_logger import DataLogger

logger = None  # will be initialized later

def init_logger(cfg):
    
    global logger
    if logger is None:
        logger = DataLogger(
            num_agents=cfg.quads_num_agents,
            #NOTE: This is hard coded !!!!
            ep_duration=1500,
            save_path=cfg.save_dir,
            num_traj=cfg.num_traj
        )
    return logger

def get_logger():
    if logger is None:
        raise RuntimeError("Logger not initialized. Did you call init_logger(cfg)?")
    return logger
