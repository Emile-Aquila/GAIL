import dataclasses
from tensordict import TensorDictBase, TensorDictParams, TensorDict
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, dispatch
from torchrl.objectives import LossModule


class Discriminator(nn.Module):
    def __init__(self, num_cells: int, obs_shape: int, act_shape: int) -> None:
        super().__init__()

        self.obs_dim = obs_shape
        self.action_dim = act_shape
        self.input_dim = self.obs_dim + self.action_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells * 2),
            nn.Tanh(),
            nn.Linear(num_cells * 2, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        input_data = torch.cat([observations, actions], dim=-1)
        logit = self.net(input_data)
        return torch.sigmoid(logit)

    def get_module(self) -> TensorDictModule:
        return TensorDictModule(self, in_keys=["observation", "action"], out_keys=["logit"])


class GAILLoss(LossModule):
    @dataclasses.dataclass
    class _AcceptedKeys:
        expert_action = "expert_action"
        expert_observation = "expert_observation"

        agent_action = "agent_action"
        agent_observation = "agent_observation"

    default_keys = _AcceptedKeys()
    out_keys = ["expert_loss", "agent_loss"]

    discriminator: TensorDictModule
    discriminator_params: TensorDictParams

    def __init__(self, discriminator: TensorDictModule, logit_eps: float = 1e-8) -> None:
        super().__init__()
        self._out_keys = None
        self._in_keys = None
        self.logit_eps: float = logit_eps

        self.convert_to_functional(
            discriminator,
            "discriminator",
            create_target_params=False,
        )

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.expert_observation,
            self.tensor_keys.expert_action,
            self.tensor_keys.agent_observation,
            self.tensor_keys.agent_action,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        expert_obs = tensordict.get(self.tensor_keys.expert_observation)
        expert_acts = tensordict.get(self.tensor_keys.expert_action)
        expert_data: TensorDict = TensorDict({"observation": expert_obs, "action": expert_acts})

        agent_obs = tensordict.get(self.tensor_keys.agent_observation)
        agent_acts = tensordict.get(self.tensor_keys.agent_action)
        agent_data: TensorDict = TensorDict({"observation": agent_obs, "action": agent_acts})

        disc_out_keys = self.discriminator.out_keys[0]
        assert self.discriminator.out_keys == [disc_out_keys]

        # discriminator loss
        with self.discriminator_params.to_module(self.discriminator):
            expert = self.discriminator(expert_data).get(disc_out_keys)

        with self.discriminator_params.to_module(self.discriminator):
            agent = self.discriminator(agent_data).get(disc_out_keys)

        expert_loss = -torch.log(expert + self.logit_eps).mean()
        agent_loss = -torch.log(1 - agent + self.logit_eps).mean()
        loss = TensorDict({"expert_loss": expert_loss, "agent_loss": agent_loss}, [])
        return loss
