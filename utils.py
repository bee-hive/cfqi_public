from typing import List, Tuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

def make_reproducible(seed, use_numpy=False, use_torch=False):
    """Set random seeds to ensure reproducibility."""
    random.seed(seed)

    if use_numpy:
        import numpy as np

        np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class NFQAgent:
    def __init__(self, nfq_net: nn.Module, optimizer: optim.Optimizer):
        self._nfq_net = nfq_net
        self._optimizer = optimizer

    def get_best_action(self, obs: np.array, unique_actions: np.array, group) -> int:
        """
        Return best action for given observation according to the neural network.
        Parameters
        ----------
        obs : np.array
            An observation to find the best action for.
        Returns
        -------
        action : int
            The action chosen by greedy selection.
        """
        concatenate_group = False
        q_list = np.zeros(len(unique_actions))
        for ii, a in enumerate(unique_actions):
            if self._nfq_net.is_compositional:
                input = torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a])], dim=0)
                q_list[ii] = self._nfq_net(input, torch.Tensor(np.asarray([[group]])))

            else:
                if not concatenate_group:
                    q_list[ii] = self._nfq_net(
                        torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a])], dim=0),
                        group * torch.ones(1),
                    )
                else:
                    q_list[ii] = self._nfq_net(
                        torch.cat([torch.FloatTensor(obs), torch.FloatTensor([a]),
                                   torch.FloatTensor([group == 0, group == 1])], dim=0),
                        group * torch.ones(1),
                    )
        return unique_actions[np.argmin(q_list)]

    def generate_pattern_set(
            self,
            rollouts: List[Tuple[np.array, int, int, np.array, bool]],
            gamma: float = 0.95,
            reward_weights=np.asarray([0.1] * 5),
            concatenate_group=False,
            groups_one=False
    ):
        """Generate pattern set.
        Parameters
        ----------
        rollouts : list of tuple
            Generated rollouts, which is a tuple of state, action, cost, next state, and done.
        gamma : float
            Discount factor. Defaults to 0.95.
        Returns
        -------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.
        """
        # _b denotes batch
        state_b, action_b, cost_b, next_state_b, done_b, group_b = zip(*rollouts)
        state_b = torch.FloatTensor(state_b)
        action_b = torch.FloatTensor(action_b)
        cost_b = torch.FloatTensor(cost_b)
        next_state_b = torch.FloatTensor(next_state_b)
        done_b = torch.FloatTensor(done_b)
        group_b = torch.FloatTensor(group_b).unsqueeze(1)

        scale_rewards = False

        if len(action_b.size()) == 1:
            action_b = action_b.unsqueeze(1)
        state_action_b = torch.cat([state_b, action_b], 1)
        # assert state_action_b.shape == (len(rollouts), state_b.shape[1] + 2) # Account for OH encoding
        if concatenate_group:
            one_hot_group = torch.nn.functional.one_hot(group_b.to(torch.int64), num_classes=2)
            state_action_b = torch.cat([state_action_b, one_hot_group.reshape(-1, 2)], 1)
        # Compute min_a Q(s', a)
        # import ipdb; ipdb.set_trace()
        next_state_left = torch.cat([next_state_b, torch.zeros(len(rollouts), 1)], 1)
        next_state_right = torch.cat([next_state_b, torch.ones(len(rollouts), 1)], 1)
        if concatenate_group:
            next_state_left = torch.cat([next_state_b, torch.zeros(len(rollouts), 1), one_hot_group.reshape(-1, 2)], 1)
            next_state_right = torch.cat([next_state_b, torch.ones(len(rollouts), 1), one_hot_group.reshape(-1, 2)], 1)
        q_next_state_left_b = self._nfq_net(
            next_state_left, group_b
        ).squeeze()
        q_next_state_right_b = self._nfq_net(
            next_state_right, group_b
        ).squeeze()

        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        with torch.no_grad():
            # wTs to replace cost_b
            if scale_rewards:
                reward_weights = np.reshape(reward_weights, (1, 5))
                reward_weights = torch.FloatTensor(reward_weights)
                s_a = torch.cat([state_b, action_b], 1)
                scaled_cost = np.matmul(reward_weights, s_a.T)
                scaled_cost = torch.FloatTensor(scaled_cost)
                target_q_values = scaled_cost.squeeze() + gamma * q_next_state_b * (
                        1 - done_b
                )
            else:
                target_q_values = cost_b.squeeze() + gamma * q_next_state_b * (
                        1 - done_b
                )
        return state_action_b, target_q_values, group_b

    def train(self, pattern_set: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Train neural network with a given pattern set.
        Parameters
        ----------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.
        Returns
        -------
        loss : float
            Training loss.
        """
        state_action_b, target_q_values, groups = pattern_set
        predicted_q_values = self._nfq_net(state_action_b, groups).squeeze()

        if self._nfq_net.is_compositional:
            if self._nfq_net.freeze_shared:
                predicted_q_values = predicted_q_values[np.where(groups == 1)[0]]
                target_q_values = target_q_values[np.where(groups == 1)[0]]
            else:
                predicted_q_values = predicted_q_values[np.where(groups == 0)[0]]
                target_q_values = target_q_values[np.where(groups == 0)[0]]
        loss = F.mse_loss(predicted_q_values, target_q_values)
        # import ipdb; ipdb.set_trace()

        # for param in self._nfq_net.parameters():
        #     loss += 10 * torch.norm(param)
        # import ipdb; ipdb.set_trace()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def evaluate(self, eval_env: gym.Env, render: bool) -> Tuple[int, str, float]:
        """Evaluate NFQ agent on evaluation environment.
        Parameters
        ----------
        eval_env : gym.Env
            Environment to evaluate the agent.
        render: bool
            If true, render environment.
        Returns
        -------
        episode_length : int
            Number of steps the agent took.
        success : bool
            True if the agent was terminated due to max timestep.
        episode_cost : float
            Total cost accumulated from the evaluation episode.
        """
        episode_length = 0
        obs = eval_env.reset()
        done = False
        render = False
        info = {"time_limit": False}
        episode_cost = 0
        while not done and not info["time_limit"]:
            action = self.get_best_action(obs, eval_env.unique_actions, eval_env.group)
            # print(action)
            obs, cost, done, info = eval_env.step(action)
            episode_cost += cost
            episode_length += 1

            if render:
                eval_env.render()

        success = (
                episode_length == eval_env.max_steps
                and abs(obs[0]) <= eval_env.x_success_range
        )

        return episode_length, success, episode_cost

class CompositionalNFQNetwork(nn.Module):
    def __init__(self, state_dim, is_compositional: bool = True, big=False, nonlinearity=nn.Sigmoid):
        super().__init__()
        self.state_dim = state_dim
#         LAYER_WIDTH = self.state_dim + 2 # Account for OH
        LAYER_WIDTH = self.state_dim + 1
        self.is_compositional = is_compositional
        self.freeze_shared = False
        self.freeze_fg = False
        if big:
            self.layers_shared = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH, LAYER_WIDTH*20),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH*20, LAYER_WIDTH * 10),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH*10, LAYER_WIDTH),
                nonlinearity(),
            )
            self.layers_fg = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH, LAYER_WIDTH*10),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH*10, LAYER_WIDTH),
                nonlinearity(),
            )
        else:
            self.layers_shared = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH, LAYER_WIDTH),
                nonlinearity()
            )
            self.layers_fg = nn.Sequential(
                nn.Linear(self.state_dim + 1, LAYER_WIDTH),
                nonlinearity(),
                nn.Linear(LAYER_WIDTH, LAYER_WIDTH),
                nonlinearity(),
            )
        self.layers_fqi = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH * 20),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 20, LAYER_WIDTH * 12),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 12, LAYER_WIDTH),
            nonlinearity(),
            )
        self.layers_last_shared = nn.Sequential(
            nn.Linear(LAYER_WIDTH, 1), nonlinearity()
        )
        self.layers_last_fg = nn.Sequential(nn.Linear(LAYER_WIDTH, 1), nonlinearity())
        self.layers_last = nn.Sequential(nn.Linear(LAYER_WIDTH * 2, 1), nonlinearity())
        # Initialize weights to [-0.5, 0.5]
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -1, 1)
        def init_weights_fg(m):
            if type(m) == nn.Linear:
                torch.nn.init.zeros_(m.weight)

        self.layers_shared.apply(init_weights)

        # if self.is_contrastive:
        self.layers_last_shared.apply(init_weights)
        self.layers_fg.apply(init_weights_fg)
        self.layers_last_fg.apply(init_weights_fg)
        self.layers_last.apply(init_weights)
        self.layers_fqi.apply(init_weights)
        if self.is_compositional:
            for param in self.layers_fg.parameters():
                param.requires_grad = False
            for param in self.layers_last_fg.parameters():
                param.requires_grad = False
        # else:
        #    self.layers_last.apply(init_weights)

    def forward(self, x: torch.Tensor, group=0) -> torch.Tensor:
        if self.is_compositional:
            x_shared = self.layers_shared(x)
            x_shared = self.layers_last_shared(x_shared)
            x_fg = self.layers_fg(x)
            x_fg = self.layers_last_fg(x_fg)
            return x_shared + torch.multiply(x_fg, group.reshape(-1, 1))
        else:
            x = self.layers_fqi(x)
            output = self.layers_last_fg(x)

            return output

    def freeze_shared_layers(self):
        for param in self.layers_shared.parameters():
            param.requires_grad = False
        for param in self.layers_last_shared.parameters():
            param.requires_grad = False

    def unfreeze_fg_layers(self):
        for param in self.layers_fg.parameters():
            param.requires_grad = True
        for param in self.layers_last_fg.parameters():
            param.requires_grad = True

    def freeze_fg_layers(self):
        for param in self.layers_fg.parameters():
            param.requires_grad = False
        for param in self.layers_last_fg.parameters():
            param.requires_grad = False

    def freeze_last_layers(self):
        for param in self.layers_last_shared.parameters():
            param.requires_grad = False
        for param in self.layers_last_fg.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self):
        for param in self.layers_last_shared.parameters():
            param.requires_grad = True
        for param in self.layers_last_fg.parameters():
            param.requires_grad = True

    def assert_correct_layers_frozen(self):

        if not self.is_compositional:
            for param in self.layers_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_shared.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == True

        elif self.freeze_shared:
            for param in self.layers_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == True
            for param in self.layers_shared.parameters():
                assert param.requires_grad == False
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == False
        else:

            for param in self.layers_fg.parameters():
                assert param.requires_grad == False
            for param in self.layers_last_fg.parameters():
                assert param.requires_grad == False
            for param in self.layers_shared.parameters():
                assert param.requires_grad == True
            for param in self.layers_last_shared.parameters():
                assert param.requires_grad == True

class MGNFQNetwork(nn.Module):
    def __init__(self, state_dim, is_compositional: bool = True, big=False, nonlinearity=nn.Sigmoid):
        super().__init__()
        self.state_dim = state_dim
        LAYER_WIDTH = self.state_dim + 1
        self.is_compositional = is_compositional
        self.freeze_shared = False
        self.freeze_fg = False
        self.layers_shared = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH*20),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH*20, LAYER_WIDTH * 10),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH*10, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_fg1 = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH*10),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH*10, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_fg2 = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH * 10),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 10, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_fg3 = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH * 10),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 10, LAYER_WIDTH),
            nonlinearity(), nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
        )
        self.layers_fqi = nn.Sequential(
            nn.Linear(self.state_dim + 1, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, LAYER_WIDTH * 20),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH*20, LAYER_WIDTH * 30),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 30, LAYER_WIDTH * 40),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 40, LAYER_WIDTH * 30),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 30, LAYER_WIDTH * 20),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 20, LAYER_WIDTH * 12),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH * 12, LAYER_WIDTH),
            nonlinearity(),
            nn.Linear(LAYER_WIDTH, 1),
            nonlinearity()
            )
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -1, 1)

        self.layers_shared.apply(init_weights)
        self.layers_fg1.apply(init_weights)
        self.layers_fg2.apply(init_weights)
        self.layers_fg3.apply(init_weights)
        self.layers_fqi.apply(init_weights)
        if is_compositional:
            for param in self.layers_fg1.parameters():
                param.requires_grad = False
            for param in self.layers_fg2.parameters():
                param.requires_grad = False
            for param in self.layers_fg3.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, group=0) -> torch.Tensor:
        if self.is_compositional:
            x_shared = self.layers_shared(x)
            x_shared = self.layers_last_shared(x_shared)
            x_fg1 = self.layers_fg(x)
            x_fg2 = self.layers_fg(x)
            x_fg3 = self.layers_fg(x)
            return x_shared + torch.multiply(x_fg1, (group == 1).reshape(-1, 1)) + torch.multiply(x_fg2, (group == 2).reshape(-1,1)) + torch.multiply(x_fg3, (group == 3).reshape(-1, 1))

        else:
            x = self.layers_fqi(x)
            return x
