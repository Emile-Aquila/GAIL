import numpy as np
import torch
from algo import Algorithm, RolloutBuffer
from model import ActorNetwork, CriticNetwork2
from pfrl.replay_buffers import ReplayBuffer
from torch import nn

from algo import Trainer
import pybullet_envs
import pybullet
import gym


class PPO(Algorithm):
    def __init__(self, state_shape, action_shape, seed=0, batch_size=512, gamma=0.995, lr_actor=3e-4, lr_critic=3e-4,
                 rollout_length=2048, num_updates=128, clip_eps=0.2, lambd=0.97, coef_ent=3e-4,
                 max_grad_norm=0.5, normalize_advantage=False):
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.buffer = RolloutBuffer(buffer_size=rollout_length, state_shape=state_shape, action_shape=action_shape)
        # rollout buffer
        self.actor = ActorNetwork(state_shape, action_shape).to(self.dev)
        self.critic = CriticNetwork2(state_shape).to(self.dev)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.batch_size = batch_size  # 1回の学習におけるmini batch数
        self.num_updates = num_updates  # 1回のupdateで学習を行う回数
        self.gamma = gamma
        self.rollout_length = rollout_length  # advantageを計算する時のrollout長
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage  # advantageの正規化をするか否か
        self.learning_steps = 0  # 学習回数

    def is_update(self, steps):  # 学習条件
        return steps % self.rollout_length == 0  # ロールアウト1回分のデータが集まるまで

    def step(self, env, state, t, steps):  # 1step進める
        action, log_pi = self.explore(state)
        n_state, rew, done, _ = env.step(action)
        if rew < -1.0:
            rew = -1.0
        done_masked = False if t == env._max_episode_steps else done  # 最大ステップ数に到達してdone=Trueになった場合を補正する.
        self.buffer.append(state=state, action=action, reward=rew, next_state=n_state, done=done_masked, log_pi=log_pi)
        if done_masked:
            t = 0
            n_state = env.reset()
        return n_state, t

    def update(self):  # 1回分の学習stepを実行する
        states, acts, rews, dones, log_pis, n_states = self.buffer.get()
        targets, advantages = self._calc_advantage(states=states, rews=rews, dones=dones,
                                                   n_states=n_states, gamma=self.gamma, lambd=self.lambd)
        loss_actors = []
        loss_critics = []

        for _ in range(self.num_updates):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)
            for i in range(0, self.rollout_length, self.batch_size):
                idxs = indices[i:i + self.batch_size]
                loss_c = self.update_critic(states[idxs], targets[idxs])
                loss_a = self.update_actor(states[idxs], acts[idxs], log_pis[idxs], advantages[idxs])
                loss_actors.append(loss_a)
                loss_critics.append(loss_c)
        self.learning_steps += 1
        return np.mean(loss_actors), np.mean(loss_critics)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        return loss_critic.cpu().detach().numpy()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        ratio = (log_pis - log_pis_old).exp_()
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        objective = torch.min(ratio * advantages, clipped_ratio * advantages).mean()  # 目的関数
        entropy = -log_pis.mean()
        loss_actor = -objective + self.coef_ent * entropy
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        return loss_actor.cpu().detach().numpy()

    def save_model(self):  # modelを保存する
        pass

    def load_weight(self, weight_path):  # modelの重みを読み込む
        pass

    def _calc_advantage(self, states, rews, dones, n_states, gamma, lambd):
        # vのtargetとGAEを計算
        with torch.no_grad():
            v_states = self.critic(states)
            v_n_states = self.critic(n_states)
        deltas = rews + gamma * (1.0 - dones) * v_n_states - v_states
        # delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
        gae = 0.0
        advantages = torch.empty_like(rews)
        for i in reversed(range(len(deltas))):
            gae = deltas[i] + gamma * lambd * (1 - dones[i]) * gae
            advantages[i] = gae
        targets = advantages + v_states
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return targets, advantages


if __name__ == '__main__':
    # ENV_ID = 'BipedalWalker-v3'
    # ENV_ID = 'Pendulum-v0'
    ENV_ID = "HalfCheetahBulletEnv-v0"
    # ENV_ID = "BipedalWalkerHardcore-v3"
    SEED = 0
    REWARD_SCALE = 1.0
    NUM_STEPS = 1 * 10 ** 6 * 3
    # NUM_STEPS = 2 * 10 ** 3
    EVAL_INTERVAL = 10 ** 3

    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)
    print("state {}".format(*env.observation_space.shape))
    print("act {}".format(*env.action_space.shape))

    state_shape = 3
    act_shape = 1

    algo = PPO(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        seed=SEED,
        normalize_advantage=True,
        clip_eps=0.2,
        coef_ent=0.05,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        seed=SEED,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
    )

    trainer.train()
    # trainer.plot()
    trainer.visualize()
