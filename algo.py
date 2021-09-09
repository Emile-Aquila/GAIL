from abc import abstractmethod
from abc import ABC
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Algorithm(ABC):
    def __init__(self):
        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def explore(self, state):  # 確率論的な行動, log(pi(a|s))を返す
        state = torch.tensor(state, dtype=torch.float, device=self.dev).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state, False)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):  # 決定論的な行動を返す
        state = torch.tensor(state, dtype=torch.float, device=self.dev).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor.sample(state, True)
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, steps):  # 学習条件
        pass

    @abstractmethod
    def step(self, env, state, t, steps):  # 1step進める
        pass

    @abstractmethod
    def update(self):  # 1回分の学習stepを実行する
        pass


class Trainer:
    def __init__(self, env, env_test, algo, seed=0, num_steps=10 ** 6, eval_interval=10 ** 4, num_eval_episodes=3):
        self.env = env
        self.env_test = env_test
        self.algo = algo

        self.env.seed(seed)
        self.env_test.seed(2 ** 31 - seed)

        self.returns = {'step': [], 'return': []}  # 平均収益を保存
        self.num_steps = num_steps  # データ収集を行うステップ数．
        self.eval_interval = eval_interval  # 評価の間のステップ数(インターバル)．
        self.num_eval_episodes = num_eval_episodes  # 評価を行うエピソード数．
        self.writer = SummaryWriter(log_dir="./logs")

    def train(self):  # num_stepsステップの間，データ収集・学習・評価を繰り返す．
        self.start_time = time()
        t = 0  # エピソードのステップ数．
        state = self.env.reset()
        for steps in tqdm(range(1, self.num_steps + 1)):
            state, t = self.algo.step(self.env, state, t, steps)

            if self.algo.is_update(steps):
                loss_critic1, loss_critic2, loss_actor, log_pis = self.algo.update()
                self.writer.add_scalar("actor loss", loss_actor, steps)
                self.writer.add_scalar("critic loss1", loss_critic1, steps)
                self.writer.add_scalar("critic loss2", loss_critic2, steps)
                self.writer.add_scalar("log pis", log_pis[0], steps)

            if steps % self.eval_interval == 0:  # 一定のインターバルで評価
                self.evaluate(steps)

    def evaluate(self, steps):  # 複数エピソード環境を動かし，平均収益を記録．
        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0
            while not done:
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        self.writer.add_scalar("rew", mean_return, steps)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {time() - self.start_time}')

    def visualize(self):  # 1エピソード環境を動かし, mp4を再生
        env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
        # env = wrap_monitor(gym.make(self.env_test))
        state = env.reset()
        done = False

        while not done:
            action = self.algo.exploit(state)
            state, rew, done, _ = env.step(action)
            env.render()
        del env

    def plot(self):  # 平均収益のグラフを描画
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()


class ReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_shape):
        self._idx = 0  # 次にデータを挿入するインデックス．
        self._size = 0  # データ数．
        self.buffer_size = buffer_size  # リプレイバッファのサイズ．

        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=self.dev)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=self.dev)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=self.dev)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=self.dev)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=self.dev)

    def append(self, state, action, reward, done, next_state):
        self.states[self._idx].copy_(torch.from_numpy(state))
        self.actions[self._idx].copy_(torch.from_numpy(action))
        self.rewards[self._idx] = float(reward)
        self.dones[self._idx] = float(done)
        self.next_states[self._idx].copy_(torch.from_numpy(next_state))

        self._idx = (self._idx + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size):
        indexes = np.random.randint(low=0, high=self._size, size=batch_size)
        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.dones[indexes],
            self.next_states[indexes]
        )


def wrap_monitor(env):
    return gym.wrappers.Monitor(env, './mp4', video_callable=lambda x: True, force=True)


def calc_log_pi(stds, noises, actions):
    #  calc : \log\pi(a|s) = \log p(u|s) - \sum_{i=1}^{|\mathcal{A}|} \log (1 - \tanh^{2}(u_i))
    #  これは, \epsilon * \sigma ~ N(0, \sigma)なる確率密度の対数を計算する関数.
    # act = tanh(\mu + \epsilon*\sigma) より, log \pi(a|s) = log p(u|s) - log (1 - tanh'(u)),  (u = \mu + \epsilon*\sigma)
    gaussian_log_probs = torch.distributions.Normal(torch.zeros_like(stds), stds).log_prob(noises).sum(dim=-1,
                                                                                                       keepdim=True)
    log_pis = gaussian_log_probs - torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis


def reparameterize(means, log_stds):
    # acts ~ N(means, stds), log_pis = f(acts), f:N(means, stds)
    stds = log_stds.exp()
    noises = stds * torch.randn_like(means)
    tmp = noises + means  # tmp ~ N(means, stds)
    acts = torch.tanh(tmp)
    log_pis = calc_log_pi(stds=stds, noises=noises, actions=acts)
    return acts, log_pis
