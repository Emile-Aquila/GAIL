import numpy as np
import torch
from algo import Algorithm, RolloutBuffer, ReplayBuffer, wrap_monitor
from model import ActorNetwork, CriticNetwork2
# from pfrl.replay_buffers import ReplayBuffer
from tqdm import tqdm
from torch import nn
from PPO import PPO
import gym
from SAC import SAC

class Discriminator(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0]+action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 1),
        )

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        # print("inputs shape {}".format(inputs.shape))
        return self.net(inputs)


class GAIL(PPO):
    def __init__(self, buffer_exp, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                 batch_size=50000, batch_size_disc=64, lr=3e-4, gamma=0.995, rollout_length=50000,
                 epoch_disc=10, epoch_ppo=50, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, lr, gamma, rollout_length,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        self.actor = PPOActor(state_shape, action_shape).to(device)
        self.critic = PPOCritic(state_shape).to(device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # デモンストレーションデータの保持するバッファ．
        self.buffer_exp = buffer_exp

        # Discriminator．
        self.disc = Discriminator(state_shape, action_shape).to(device)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(), lr=lr)

        self.batch_size_disc = batch_size_disc
        self.epoch_disc = epoch_disc

    def update(self):
        # GAILでは，環境からの報酬情報は用いない．
        states, actions, _, dones, log_pis = self.buffer.get()

        # Discriminatorの学習．
        for _ in range(self.epoch_disc):
            idxes = np.random.randint(low=0, high=self.rollout_length, size=self.batch_size_disc)
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size_disc)[:2]
            self.update_disc(states[idxes], actions[idxes], states_exp, actions_exp)

        with torch.no_grad():
            rewards = - torch.log(torch.nn.functional.sigmoid(self.disc(states[:-1], actions)))
        # PPOの学習．
        self.update_ppo(states, actions, rewards, dones, log_pis)

    def update_disc(self, states, actions, states_exp, actions_exp):
        loss_pi = -torch.log(torch.nn.functional.sigmoid(self.disc(states, actions))).mean()
        loss_exp = -torch.log(torch.nn.functional.sigmoid(-self.disc(states_exp, actions_exp))).mean()
        self.optim_disc.zero_grad()
        (loss_pi + loss_exp).backward()
        self.optim_disc.step()


def run_agent(env_id, agent, replay_buffer, step_time, p_rand=0.0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # env = wrap_monitor(gym.make(env_id))
    env = gym.make(env_id)
    state = env.reset()
    for _ in tqdm(range(step_time)):
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action, _ = agent.explore(torch.tensor(state, dtype=torch.float, device=device))
            # action = add_random_noise(action, std)
        n_state, reward, done, _ = env.step(action)
        replay_buffer.append(state, action, reward, n_state, done)
        if done:
            state = env.reset()
        else:
            state = n_state


if __name__ == '__main__':
    ENV_ID = "HalfCheetahBulletEnv-v0"
    env = gym.make(ENV_ID)
    env_online = gym.make(ENV_ID)
    state_shape = 3
    act_shape = 1

    agent = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    )
    weight_path_actor = "/home/emile/Documents/Code/GAIL/models/actor.pth"
    weight_path_critic = "/home/emile/Documents/Code/GAIL/models/critic.pth"
    buffer_length = 10**5
    step_time = 10**5
    agent.load_weight(weight_path_actor, weight_path_critic)
    replay_buffer = ReplayBuffer(buffer_length, env.observation_space.shape, env.action_space.shape)
    run_agent(ENV_ID, agent, replay_buffer, step_time, p_rand=0.1)
    # ここまでデータ収集

    BATCH_SIZE = 1000  # PPOの学習バッチサイズ．
    ROLLOUT_LENGTH = 1000  # ロールアウトの長さ．
    EPOCH_DISC = 10  # ロールアウト毎のPPOの学習エポック数．
    EPOCH_PPO = 200  # ロールアウト毎のDiscriminatorの学習エポック数．
    NUM_STEPS = 50000
    EVAL_INTERVAL = 2500
    SEED = 0

    algo = GAIL(
        buffer_exp=replay_buffer,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        seed=SEED,
        batch_size=BATCH_SIZE,
        rollout_length=ROLLOUT_LENGTH,
        epoch_disc=EPOCH_DISC,
        epoch_ppo=EPOCH_PPO
    )
    trainer = GAILTrainer(
        env=env,
        env_online=env_online,
        algo=algo,
        seed=SEED,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL
    )
    trainer.train()
