from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_neighbor_encoder_type", ["mean_embed"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 '
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=corl '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=final_model_architecture'
)

_experiment = Experiment(
    "final_model_architecture",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])