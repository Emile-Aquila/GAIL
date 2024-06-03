from tensordict import TensorDict
from torchrl.modules import ProbabilisticActor, ValueOperator

from gail import Discriminator, GAILLoss
from torchrl.envs.libs.gym import GymEnv
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter
from torchrl.objectives import KLPENPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs
from simple_ppo import get_network
from tqdm import tqdm
import os


def pretrain_gail(policy_module: ProbabilisticActor, value_module: ValueOperator, expert_data: ReplayBuffer) -> None:
    # params
    discriminator_cells = 256
    num_updates = 50
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

    obs_shape = env.observation_spec["observation"].shape[-1]
    act_shape = env.action_spec.shape[-1]
    check_env_specs(env)

    # Advantage
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    # PPO Module
    loss_module = KLPENPPOLoss(
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

    replay_buffer_ppo = ReplayBuffer(
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
    for i, tensordict_data in tqdm(enumerate(collector)):  # num_updates回分回す
        data_view = tensordict_data.clone().reshape(-1)
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

        # PPO
        tensordict_data_ppo = tensordict_data.clone()
        tensordict_data_ppo["next"]["reward"] = disc(tensordict_data["observation"], tensordict_data["action"])

        with torch.no_grad():
            advantage_module(tensordict_data_ppo)
        data_view = tensordict_data_ppo.reshape(-1)
        replay_buffer_ppo.extend(data_view.cpu())

        for j in range(frames_per_batch // gail_step_size):
            subdata = replay_buffer_ppo.sample(gail_step_size)
            loss_ppo = loss_module(subdata)
            loss_value = (
                    loss_ppo["loss_objective"]
                    + loss_ppo["loss_critic"]
                    + loss_ppo["loss_entropy"]
            )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim_ppo.step()
            optim_ppo.zero_grad()

    # mkdir
    if not os.path.exists("datas"):
        os.makedirs("datas")

    torch.save(policy_module.module.state_dict(), "./datas/policy_gail.pth")
    torch.save(value_module.state_dict(), "./datas/value_gail.pth")


def get_dataset(env: TransformedEnv, data_size: int) -> ReplayBuffer:
    policy_module, value_module = get_network(
        64,
        env.action_spec,
    )
    policy_module.module.load_state_dict(torch.load("datas/policy.pth"))
    value_module.load_state_dict(torch.load("datas/value.pth"))

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


def test_model(policy_module: ProbabilisticActor, env: TransformedEnv) -> None:
    trial_times = 100
    ave_reward = 0
    ave_steps = 0
    for i in range(trial_times):
        tmp = env.rollout(1000, policy_module)
        ave_reward += tmp["next"]["reward"].sum().item()
        ave_steps += tmp["next"]["reward"].shape[0]
    print("total reward:", ave_reward / trial_times)
    print("average steps:", ave_steps / trial_times)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

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
        64,
        env.action_spec,
    )

    # Setting Actor, Critic
    policy_module(env.reset())
    value_module(env.reset())
    print("network initialized")

    expert_data = get_dataset(env, 100000)
    expert_data.dumps("expert_data.pth")
    print("save expert data")

    print("start GAIL training")
    pretrain_gail(policy_module, value_module, expert_data)

    print("start testing")
    policy_module2, _ = get_network(64, env.action_spec)
    policy_module2.module.load_state_dict(torch.load("datas/policy.pth"))
    test_model(policy_module2, env)

    test_model(policy_module, env)

    policy_module.module.load_state_dict(torch.load("./datas/policy_gail.pth"))
    test_model(policy_module, env)
