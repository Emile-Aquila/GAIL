import torch
from tensordict import TensorDict
from gail import Discriminator, GAILLoss
from torchrl.envs.libs.gym import GymEnv
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs
from simple_ppo import get_network
from tqdm import tqdm


def pretrain_gail(policy_module, value_module, expert_data: ReplayBuffer):
    # params
    discriminator_cells = 256
    num_updates = 1000
    gail_step_size = 128

    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-3
    lr = 3e-4
    max_grad_norm = 1.0
    frames_per_batch = 1000
    total_frames = 50_000

    # Setting Envs
    base_env = GymEnv("InvertedDoublePendulum-v4")
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)

    obs_shape = env.observation_spec["observation"].shape[-1]
    act_shape = env.action_spec.shape[-1]

    print(env.observation_spec.shape)
    check_env_specs(env)

    # Advantage
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    # PPO Module
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim_ppo = torch.optim.Adam(loss_module.parameters(), lr)

    # replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement()
    )

    # Setting Data Collector
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * num_updates,
        split_trajs=False,
    )

    # Setting GAIL
    disc = Discriminator(discriminator_cells, obs_shape, act_shape).to(torch.device("cpu"))
    discriminator: TensorDictModule = disc.get_module()
    gail_loss: GAILLoss = GAILLoss(discriminator)
    optim_gail = torch.optim.Adam(gail_loss.parameters())

    # training loop
    for i, tensordict_data in enumerate(collector):  # num_updates回分回す
        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        for j in range(gail_step_size):
            data = expert_data.sample(frames_per_batch)
            agent_data = replay_buffer.sample(frames_per_batch)

            input_dict: TensorDict = TensorDict({
                "expert_observation": data["observation"],
                "expert_action": data["action"],
                "agent_observation": agent_data["observation"],
                "agent_action": agent_data["action"],
            })

            losses = gail_loss(input_dict)
            loss = losses["expert_loss"] + losses["agent_loss"]
            optim_gail.zero_grad()
            loss.backward()
            optim_gail.step()

        loss_ppo = loss_module(data_view)
        loss_value = (
                loss_ppo["loss_objective"]
                + loss_ppo["loss_critic"]
                + loss_ppo["loss_entropy"]
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        optim_ppo.step()
        optim_ppo.zero_grad()


def get_dataset(env, data_size: int) -> ReplayBuffer:
    policy_module, value_module = get_network(
        256,
        env.action_spec,
    )
    policy_module.module.load_state_dict(torch.load("policy.pth"))
    value_module.module.load_state_dict(torch.load("value.pth"))

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=data_size,
        total_frames=1,
        split_trajs=False,
    )

    rb: ReplayBuffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=data_size),
        sampler=SamplerWithoutReplacement()
    )

    for i, tensordict_data in enumerate(collector):
        rb.extend(tensordict_data)

    return rb


if __name__ == "__main__":
    base_env = GymEnv("InvertedDoublePendulum-v4")
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    policy_module, value_module = get_network(
        256,
        env.action_spec,
    )
    expert_data = get_dataset(env, 50000)

    pretrain_gail(policy_module, value_module, expert_data)
