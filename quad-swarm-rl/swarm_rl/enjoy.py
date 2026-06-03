import sys

from sample_factory.enjoy import enjoy

## edited sample factory enjoy
# from local_sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
from swarm_rl.env_wrappers.logger_manager import init_logger

def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=True)
    init_logger(cfg)
    status = enjoy(cfg)
    if isinstance(status, tuple):
        return int(status[0])
    return status

if __name__ == '__main__':
    sys.exit(main())
