from collections import defaultdict
from torchrl.envs.libs.gym import GymEnv
import torch
import torch.nn as nn
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, TensorSpec
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter
from torchrl.modules import ValueOperator, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, set_exploration_type, ExplorationType
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def get_network(num_cells: int, act_spec: TensorSpec):
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(act_spec.shape[-1]*2),
        NormalParamExtractor(),
    )
    policy_dict = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    policy_module = ProbabilisticActor(
        module=policy_dict,
        spec=act_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": act_spec.space.low,
            "max": act_spec.space.high,
        },
        default_interaction_type=ExplorationType.RANDOM,
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(num_cells),
        nn.Tanh(),
        nn.LazyLinear(1),
    )
    value_module = ValueOperator(module=value_net, in_keys=["observation"])

    return policy_module, value_module


if __name__ == "__main__":
    num_cells = 128
    sub_batch_size = 64
    num_epochs = 10
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 3e-4
    lr = 3e-4
    max_grad_norm = 1.0
    frames_per_batch = 5000
    total_frames = 100000

    # Setting Envs
    base_env = GymEnv("BipedalWalker-v3")
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    # Setting Actor Critic
    policy_module, value_module = get_network(num_cells, env.action_spec)
    policy_module(env.reset())
    value_module(env.reset())
    print("networks initialized")

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement()
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    check_env_specs(env)

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        for _ in range(num_epochs):
            with torch.no_grad():
                advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata)
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 5 == 0:
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                eval_rollout = env.rollout(3000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        scheduler.step()

    # mkdir
    if not os.path.exists("./datas"):
        os.makedirs("./datas")

    torch.save(policy_module.module.state_dict(), "./datas/policy.pth")
    torch.save(value_module.state_dict(), "./datas/value.pth")

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

