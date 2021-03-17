import torch
import torch.nn.functional as F

import config as cfg


class Agent:
    """
    Generic Agent that has a buffer to store samples,
    a policy to act and a model to compute Q.
    """

    def __init__(self, model, policy, buffer):
        self.model = model
        self.policy = policy
        self.buffer = buffer

    def act(self, state):
        """Compute action according to policy for given state"""
        return self.policy.action(state)

    def fill_buffer(self, episode):
        """Appends an episode to the buffer"""
        self.buffer.append(episode)

    def update(self):
        """
        Compute loss based on buffer sample, update model weights and adjust buffer probabilities.
        """
        batch = self.buffer.get_sample()
        states, actions, rewards, dones, next_states = zip(*batch)

        self.model.optimizer.zero_grad()

        states = torch.tensor(states).float().to(self.model.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.model.device)
        rewards = torch.tensor(rewards).float().to(self.model.device)
        dones = torch.tensor(dones).float().to(self.model.device)
        next_states = torch.tensor(next_states).float().to(self.model.device)

        q_values = (
            torch.gather(self.model(states), dim=-1, index=actions)
            .squeeze()
            .to(self.model.device)
        )

        target_q_values = (
            rewards
            + (1 - dones)
            * self.model.discount
            * self.model(next_states).max(dim=-1)[0].detach()
        )

        targets = (
            F.mse_loss(q_values, target_q_values, reduction="none").cpu().detach().numpy()
        )
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        self.model.optimizer.step()

        self.buffer.update_weights(targets)

        return loss.item()


class DDQNAgent:
    """
    Generic DoubleDQN Agent that has a buffer to store samples,
    a policy to act and two models to compute/evaluate Q.
    """

    def __init__(self, actor_model, eval_model, policy, buffer):
        self.actor_model = actor_model
        self.eval_model = eval_model
        self.policy = policy
        self.buffer = buffer

    def act(self, state):
        """Compute action according to policy for given state"""
        return self.policy.action(state)

    def fill_buffer(self, episode):
        """Appends an episode to the buffer"""
        self.buffer.append(episode)

    def sync_nets(self, tau=cfg.TAU):
        """Polyak averaging"""
        for target_param, local_param in zip(
            self.eval_model.parameters(), self.actor_model.parameters()
        ):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def update(self):
        """
        Compute (doubleDQN) loss based on buffer sample,
        update model weights and adjust buffer probabilities.
        """
        batch = self.buffer.get_sample()
        states, actions, rewards, dones, next_states = zip(*batch)

        self.actor_model.optimizer.zero_grad()

        states = torch.tensor(states).float().to(self.actor_model.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.actor_model.device)
        rewards = torch.tensor(rewards).float().to(self.actor_model.device)
        dones = torch.tensor(dones).float().to(self.actor_model.device)
        next_states = torch.tensor(next_states).float().to(self.actor_model.device)

        q_values = (
            torch.gather(self.actor_model(states), dim=-1, index=actions)
            .squeeze()
            .to(self.actor_model.device)
        )

        target_q_values = (
            rewards
            + (1 - dones)
            * self.actor_model.discount
            * self.eval_model(next_states).max(dim=-1)[0].detach()
        )

        targets = (
            F.mse_loss(q_values, target_q_values, reduction="none").cpu().detach().numpy()
        )
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        self.actor_model.optimizer.step()

        self.buffer.update_weights(targets)

        return loss.item()
