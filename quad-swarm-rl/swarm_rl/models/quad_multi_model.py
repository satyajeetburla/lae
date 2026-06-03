import json
from pathlib import Path

import torch
from torch import nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import fc_layer, nonlinearity

from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE

from swarm_rl.env_wrappers.logger_manager import get_logger
from swarm_rl.models.attention_layer import MultiHeadAttention, OneHeadAttention

from sample_factory.model.model_utils import model_device


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAE_CONFIG = REPO_ROOT / "artifacts" / "lae" / "paper_h250_m10" / "config.json"


class NullLatentLogger:
    def save_latent(self, *_args, **_kwargs):
        return None


def repo_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def artifact_path(path_value, config_path):
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


class QuadNeighborhoodEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs


class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


class QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding


class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding


# Paper LAE GRU latent collision model.
class LatentPredictorGRU(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, seq):
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.head(last)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev,     h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QuadMultiHeadAttentionEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.is_critic = cfg.is_critic
        # Internal params
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )

        num_heads = 4
        # # Attention Layer
        self.attention_layer = MultiHeadAttention(num_heads, cfg.rnn_size, cfg.rnn_size, cfg.rnn_size)
        # self.attention_layer = OneHeadAttention(cfg.rnn_size)

        # MLP Layer
        self.encoder_output_size = 2 * cfg.rnn_size
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())

        try:
            self.logger = get_logger()
        except RuntimeError:
            self.logger = NullLatentLogger()

        self.mode = int(getattr(cfg, "sae_encoder_mode", 0) or 0)
        self.use_mask = True
        self.head_dim = 10
        self.edit_dim = 20
        self.classifier = None
        self.latent_coll_model = None
        self.window = 3
        self.tail_buffer = []
        self.mask_count = None

        if self.mode not in (0, 13):
            raise ValueError(
                f"Unsupported sae_encoder_mode={self.mode}. "
                "This cleaned ICRA branch supports mode 0 (no edit) and "
                "mode 13 (paper GRU LCWM) only."
            )
        if self.mode == 13:
            self._load_lae_gru_config(cfg)

    def _load_lae_gru_config(self, cfg):
        config_path = repo_path(getattr(cfg, "lae_config", None)) or DEFAULT_LAE_CONFIG
        if not config_path.exists():
            raise FileNotFoundError(
                f"LAE config not found: {config_path}. "
                "Pass --lae_config or place the paper config at "
                "artifacts/lae/paper_h250_m10/config.json."
            )

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        classifier_cfg = config.get("classifier", {})
        editor_cfg = config.get("editor", {})
        classifier_path = artifact_path(classifier_cfg.get("path"), config_path)
        editor_path = artifact_path(editor_cfg.get("path"), config_path)

        missing = [
            str(path)
            for path in (classifier_path, editor_path)
            if path is None or not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing LAE artifact(s): "
                + ", ".join(missing)
                + ". Keep artifact paths relative to the LAE config."
            )

        self.classifier_dim = int(classifier_cfg.get("input_dim", 30))
        encoder_embedding_dim = 3 * int(getattr(cfg, "rnn_size"))
        if encoder_embedding_dim != self.classifier_dim:
            raise ValueError(
                "Mode 13 LAE classifier expects "
                f"{self.classifier_dim} input features, but this policy encoder "
                f"produces {encoder_embedding_dim} features. Use the paper "
                "setting --rnn_size=10 or provide a matching LAE config."
            )
        self.hidden_clas = list(classifier_cfg.get("hidden_dims", [512, 512, 256, 128]))
        classifier_classes = int(classifier_cfg.get("num_classes", 2))
        self.classifier = Classifier(
            self.classifier_dim,
            hidden_dims=self.hidden_clas,
            num_classes=classifier_classes,
        )
        classifier_state = torch.load(classifier_path, map_location="cpu")
        classifier_state = classifier_state.get("state_dict", classifier_state)
        self.classifier.load_state_dict(classifier_state, strict=True)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.to(model_device(self))

        self.window = int(editor_cfg.get("window", 3))
        latent_dim = int(editor_cfg.get("latent_dim", 20))
        self.edit_dim = latent_dim
        self.head_dim = self.classifier_dim - self.edit_dim
        if self.head_dim <= 0:
            raise ValueError(
                f"Invalid LAE dimensions: classifier input {self.classifier_dim}, "
                f"latent_dim {self.edit_dim}."
            )
        hidden_dim = int(editor_cfg.get("hidden_dim", 256))
        num_layers = int(editor_cfg.get("num_layers", 4))
        dropout = float(editor_cfg.get("dropout", 0.2))
        model_type = str(editor_cfg.get("type", "gru")).lower()
        if model_type != "gru":
            raise ValueError(f"Unsupported LAE editor type '{model_type}'. Only 'gru' is supported.")

        self.latent_coll_model = LatentPredictorGRU(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        editor_state = torch.load(editor_path, map_location="cpu")
        editor_state = editor_state.get("model_state", editor_state.get("state_dict", editor_state))
        self.latent_coll_model.load_state_dict(editor_state, strict=True)
        self.latent_coll_model.eval()
        for param in self.latent_coll_model.parameters():
            param.requires_grad = False
        self.latent_coll_model.to(model_device(self))
        self.tail_buffer = []
        self.mask_count = None


    def forward(self, obs_dict):
        obs = obs_dict['obs']
        batch_size = obs.shape[0]
        obs_self = obs[:, :self.self_obs_dim]
        obs_neighbor = obs[:, self.self_obs_dim: self.self_obs_dim + self.all_neighbor_obs_dim]
        obs_obstacle = obs[:, self.self_obs_dim + self.all_neighbor_obs_dim:]

        self_embed = self.self_embed_layer(obs_self)
        # print("self_embed", self_embed)
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor)
        # print("neighbor_embed", neighbor_embed)

        obstacle_embed = self.obstacle_embed_layer(obs_obstacle)
        # print("obstacle_embed", obstacle_embed)

        neighbor_embed = neighbor_embed.view(batch_size, 1, -1)
        obstacle_embed = obstacle_embed.view(batch_size, 1, -1)
        attn_embed = torch.cat((neighbor_embed, obstacle_embed), dim=1)

        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        attn_embed = attn_embed.view(batch_size, -1)

        embeddings = torch.cat((self_embed, attn_embed), dim=1)
        if not self.is_critic:
            embeddings, mask = self._apply_mode(embeddings)
            self.logger.save_latent(embeddings)

        out = self.feed_forward(embeddings)
        # if not self.is_critic:
        #     out, mask = self._apply_mode(out)
        #     self.logger.save_latent(out)

        return out


    def get_out_size(self):
        return self.encoder_output_size

    def _apply_mode(self, emb):
        if self.mode == 0 or not self.use_mask:
            return emb, torch.zeros(emb.size(0), dtype=torch.bool, device=emb.device)
        if self.mode != 13:
            raise ValueError(
                f"Unsupported sae_encoder_mode={self.mode}. "
                "This cleaned ICRA branch supports only mode 0 and mode 13."
            )
        if emb.shape[1] != self.classifier_dim:
            raise ValueError(
                f"Mode 13 LAE classifier expects embeddings with {self.classifier_dim} "
                f"features, got {emb.shape[1]}."
            )
        head, tail = emb[:, :self.head_dim], emb[:, self.head_dim:]
        assert tail.shape[1] == self.edit_dim, f"Tail shape mismatch: expected {self.edit_dim}, got {tail.shape[1]}"

        with torch.no_grad():
            self.classifier.eval()
            preds = self.classifier(emb).argmax(dim=1)
            unsafe = preds == 1
        return self._latent_collison_model(head, tail, unsafe)

    def _latent_collison_model(self, head, tail, mask):
        num_agents, _ = tail.shape

        if (
            self.mask_count is None
            or self.mask_count.device != tail.device
            or self.mask_count.numel() != num_agents
        ):
            self.mask_count = torch.zeros(num_agents, dtype=torch.int, device=tail.device)
            self.tail_buffer = []

        self.mask_count = torch.where(mask, self.mask_count + 1, torch.zeros_like(self.mask_count))

        self.tail_buffer.append(tail)
        if len(self.tail_buffer) > self.window:
            self.tail_buffer.pop(0)

        out_tail = tail.clone()
        if len(self.tail_buffer) == self.window:
            valid = self.mask_count >= self.window
            if valid.any():
                seq = torch.stack(self.tail_buffer, dim=1)
                out_tail[valid] = self.latent_coll_model(seq[valid])

        return torch.cat([head, out_tail], dim=1), mask


class QuadMultiHeadAttentionEncoder_Sim2Real(QuadMultiHeadAttentionEncoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)

        # Internal params
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )

        # Attention Layer
        self.attention_layer = OneHeadAttention(cfg.rnn_size)

        self.hidden_size = cfg.rnn_size

        # MLP Layer
        self.encoder_output_size = 3 * cfg.rnn_size
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())


class QuadMultiEncoder(Encoder):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        self.use_obstacles = cfg.quads_use_obstacles

        # Neighbor
        neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        if cfg.quads_neighbor_obs_type == 'none':
            num_use_neighbor_obs = 0
        else:
            if cfg.quads_neighbor_visible_num == -1:
                num_use_neighbor_obs = cfg.quads_num_agents - 1
            else:
                num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.all_neighbor_obs_size = neighbor_obs_dim * num_use_neighbor_obs

        # # Neighbor Encoder
        neighbor_encoder_out_size = 0
        self.neighbor_encoder = None

        if num_use_neighbor_obs > 0:
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            if neighbor_encoder_type == 'mean_embed':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'attention':
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'mlp':
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'no_encoder':
                # Blind agent
                self.neighbor_encoder = None
            else:
                raise NotImplementedError

        if self.neighbor_encoder:
            neighbor_encoder_out_size = neighbor_hidden_size

        fc_encoder_layer = cfg.rnn_size
        # Encode Self Obs
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # Encode Obstacle Obs
        obstacle_encoder_out_size = 0
        if self.use_obstacles:
            obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
            obstacle_hidden_size = cfg.quads_obst_hidden_size
            self.obstacle_encoder = nn.Sequential(
                fc_layer(obstacle_obs_dim, obstacle_hidden_size),
                nonlinearity(cfg),
                fc_layer(obstacle_hidden_size, obstacle_hidden_size),
                nonlinearity(cfg),
            )
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (obstacle_obs_dim,))

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # This is followed by another fully connected layer in the action parameterization, so we add a nonlinearity
        # here
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, 2 * cfg.rnn_size),
            nn.Tanh(),
        )

        self.encoder_out_size = 2 * cfg.rnn_size

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        self_embed = self.self_encoder(obs_self)
        embeddings = self_embed
        batch_size = obs_self.shape[0]
        # Relative xyz and vxyz for the Entire Minibatch (batch dimension is batch_size * num_neighbors)
        if self.neighbor_encoder:
            neighborhood_embedding = self.neighbor_encoder(obs_self, obs, self.all_neighbor_obs_size, batch_size)
            embeddings = torch.cat((embeddings, neighborhood_embedding), dim=1)

        if self.use_obstacles:
            obs_obstacles = obs[:, self.self_obs_dim + self.all_neighbor_obs_size:]
            obstacle_embeds = self.obstacle_encoder(obs_obstacles)
            embeddings = torch.cat((embeddings, obstacle_embeds), dim=1)

        out = self.feed_forward(embeddings)
        return out

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    if cfg.quads_encoder_type == "attention":
        if cfg.quads_sim2real:
            model = QuadMultiHeadAttentionEncoder_Sim2Real(cfg, obs_space)
        else:
            model = QuadMultiHeadAttentionEncoder(cfg, obs_space)
    else:
        model = QuadMultiEncoder(cfg, obs_space)
    return model


def register_models():
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)
