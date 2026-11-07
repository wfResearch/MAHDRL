"""Candidate scoring and fixed-slot dueling networks."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(512, 256), head_hidden_dims=None):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.feature = nn.Sequential(*layers)
        last_h = dims[-1]
        if head_hidden_dims is None:
            head_hidden_dims = (last_h, last_h)
        self.adv = self._mlp(last_h, head_hidden_dims, output_dim)
        self.val = self._mlp(last_h, head_hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        a = self.adv(h)
        v = self.val(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    @staticmethod
    def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)

def _polyak_update(target: nn.Module, online: nn.Module, tau: float):
    """target = (1 - tau) * target + tau * online"""
    with torch.no_grad():
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)
from mahdrl.config import ENTITY
from mahdrl.config import TRUNK
import numpy as np
import torch
from torch import nn

def mlp(dimensions):
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def copy_observation(obs):
    return {key: np.array(obs[key], dtype=np.float32, copy=True) for key in ('context', 'candidates', 'exists_mask')}

def validate_observation(obs, mask, *, decision=True):
    context, candidates, exists = (np.asarray(obs[k]) for k in ('context', 'candidates', 'exists_mask'))
    mask = np.asarray(mask)
    if context.ndim != 1 or candidates.ndim != 2 or exists.ndim != 1 or (mask.shape != exists.shape) or (len(candidates) != len(exists)):
        raise ValueError('Invalid set observation/mask shapes')
    if not all((np.isfinite(x).all() for x in (context, candidates, exists, mask))):
        raise ValueError('Non-finite set observation')
    if not all((np.isin(x, (0, 1)).all() for x in (exists, mask))):
        raise ValueError('Set masks must be binary')
    if np.any(mask > exists):
        raise ValueError('Padding cannot be a legal action')
    if decision and (not np.any(mask > 0.5)):
        raise ValueError('No legal action at a non-terminal decision')

def collate(observations, masks, device):
    n = max(1, max((len(x['exists_mask']) for x in observations)))
    b, f = (len(observations), observations[0]['candidates'].shape[1])
    features = np.zeros((b, n, f), np.float32)
    exists = np.zeros((b, n), np.float32)
    legal = np.zeros_like(exists)
    for i, (obs, mask) in enumerate(zip(observations, masks)):
        validate_observation(obs, mask, decision=False)
        k = len(mask)
        features[i, :k], exists[i, :k], legal[i, :k] = (obs['candidates'], obs['exists_mask'], mask)
    return tuple((torch.as_tensor(x, dtype=torch.float32, device=device) for x in (np.stack([x['context'] for x in observations]), features, exists, legal)))

class CandidateDuelingQNet(nn.Module):

    def __init__(self, context_dim, candidate_dim, hidden_dims=TRUNK, entity_dims=ENTITY):
        super().__init__()
        self.context_dim, self.candidate_dim = (context_dim, candidate_dim)
        self.entity = mlp((candidate_dim, *entity_dims))
        self.entity_activation = nn.ReLU()
        e, h = (entity_dims[-1], hidden_dims[-1])
        self.feature = mlp((context_dim + 2 * e + 2, *hidden_dims))
        self.feature_activation = nn.ReLU()
        self.val = mlp((h, h, h, 1))
        self.adv = mlp((h + e, h, h, 1))

    def forward(self, context, candidates, exists_mask, action_mask):
        if context.shape[-1] != self.context_dim or candidates.shape[-1] != self.candidate_dim:
            raise ValueError('Candidate feature schema mismatch')
        e = self.entity_activation(self.entity(candidates))
        exists, legal = (exists_mask > 0.5, action_mask > 0.5)
        n = exists.sum(1, keepdim=True)
        mean = (e * exists.unsqueeze(-1)).sum(1) / n.clamp_min(1)
        maximum = e.masked_fill(~exists.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(n > 0, maximum, torch.zeros_like(maximum))
        counts = torch.cat((n, legal.sum(1, keepdim=True)), 1).to(e.dtype).log1p()
        h = self.feature_activation(self.feature(torch.cat((context, mean, maximum, counts), 1)))
        a = self.adv(torch.cat((h.unsqueeze(1).expand(-1, e.shape[1], -1), e), -1)).squeeze(-1)
        center = (a * legal).sum(1, keepdim=True) / legal.sum(1, keepdim=True).clamp_min(1)
        return self.val(h) + a - center
from mahdrl.config import TRUNK
from mahdrl.config import VM_CAPACITY
from mahdrl.config import VM_INPUT_DIM

class FixedVMDuelingQNet(DuelingQNet):

    def __init__(self, hidden_dims=TRUNK):
        super().__init__(VM_INPUT_DIM, VM_CAPACITY, hidden_dims)

    def forward(self, x, mask):
        if x.ndim != 2 or x.shape[1] != VM_INPUT_DIM or mask.shape != (x.shape[0], VM_CAPACITY):
            raise ValueError('Fixed VA network shape mismatch')
        h = self.feature(x)
        advantage = self.adv(h)
        legal = (mask > 0.5).to(advantage.dtype)
        center = (advantage * legal).sum(1, keepdim=True) / legal.sum(1, keepdim=True).clamp_min(1)
        return self.val(h) + advantage - center
