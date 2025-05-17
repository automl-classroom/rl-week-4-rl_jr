"""
Deep Q-Learning implementation.
"""

import sys
import os
# Otherwise I get a ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q-network and frozen target
        self.q = QNetwork(obs_dim, n_actions)
        self.target_q = QNetwork(obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # TODO: implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε

        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-self.total_steps / self.epsilon_decay)

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            with torch.no_grad():
                qvals = self.q.forward(torch.tensor(state, dtype=torch.float32))  # noqa: F841

            action = np.argmax(qvals.detach().numpy())
        else:
            if np.random.rand() < self.epsilon():
                # TODO: sample random action
                action = np.random.choice(self.env.action_space.n)
            else:
                # TODO: select purely greedy action from Q(s)
                qvals = self.q.forward(torch.tensor(state, dtype=torch.float32))
                action = np.argmax(qvals.detach().numpy())

        return int(action)

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32)  # noqa: F841

        # # TODO: pass batched states through self.q and gather Q(s,a)
        pred = self.q.forward(s).gather(1, a).squeeze()

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            # Target = 𝑟 + 𝛾 max𝑎′̂ 𝑄(𝑠′, 𝑎′; w)
            target_qvals = self.target_q.forward(s_next)
            max_a = target_qvals.max(1)[0]  # max(dim) -> (values, indices)
            target = r + self.gamma * max_a

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        mean_rewards: List[float] = []
        frames: List[int] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample a batch from replay buffer
                batch = self.buffer.sample(batch_size=self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % eval_interval == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = sum(recent_rewards[-eval_interval:]) / eval_interval
                    mean_rewards.append(avg)
                    frames.append(frame)
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")
        return mean_rewards, frames, recent_rewards


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = DQNAgent(
        env                = env, 
        buffer_capacity    = cfg.agent.buffer_capacity,
        batch_size         = cfg.agent.batch_size,
        lr                 = cfg.agent.learning_rate,
        gamma              = cfg.agent.gamma,
        epsilon_start      = cfg.agent.epsilon_start,
        epsilon_final      = cfg.agent.epsilon_final,
        epsilon_decay      = cfg.agent.epsilon_decay,
        target_update_freq = cfg.agent.target_update_freq,
        seed               = cfg.seed
    )

    mean_rewards, frames, recent_rewards = agent.train(num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval)

    # plot for Level 1
    if False:
        linear_layers = [l for l in agent.q.modules() if isinstance(l, nn.Linear)]

        depth = len(linear_layers)
        width = linear_layers[0].out_features
        size_replay_buffer = int(cfg.agent.buffer_capacity)
        batch_size = int(cfg.agent.batch_size)

        plt.plot(frames, mean_rewards, label='Episode reward')
        plt.xlabel('Number of frames')
        plt.ylim(bottom=0)
        plt.ylabel('Mean reward')
        title = f'Architecture: {depth=}; {width=}; {size_replay_buffer=}; {batch_size=}'
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"../../../rl_exercises/week_4/plots/{title.replace(': ', '-').replace('=', '').replace('; ', '-').lower()}.png", bbox_inches='tight')
        if False:
            plt.show()

    # store data for for Level 2
    if False:
        seed = cfg.seed
        data = {
            'seed': seed,
            'mean_rewards': mean_rewards,
            'frames': frames,
            'recent_rewards': recent_rewards
        }
        with open(f"../../../rl_exercises/week_4/results_level_2/seed_{seed}.json", 'w') as file:
            json.dump(data, file)



if __name__ == "__main__":
    main()
