import torch
import torch.nn.functional as F


class Agent:
    def __init__(self, model, policy, buffer):
        self.model = model
        self.policy = policy
        self.buffer = buffer

    def act(self, state):
        return self.policy.action(state)

    def fill_buffer(self, episode):
        self.buffer.append(episode)

    def update(self):
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

        targets = F.mse_loss(q_values, target_q_values, reduction="none").cpu().detach().numpy()
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        self.model.optimizer.step()

        self.buffer.update_weights(targets)

        return loss.item()


class DDQNAgent:
    def __init__(self, actor_model, eval_model, policy, buffer):
        self.actor_model = actor_model
        self.eval_model = eval_model
        self.policy = policy
        self.buffer = buffer
        self.updates = 0  # how often update function was called

    def act(self, state):
        return self.policy.action(state)

    def fill_buffer(self, episode):
        self.buffer.append(episode)

    def sync_nets(self, tau=0.01):
        for target_param, local_param in zip(
            self.eval_model.parameters(), self.actor_model.parameters()
        ):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def update(self):
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

        targets = F.mse_loss(q_values, target_q_values, reduction="none").detach().numpy()
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        self.actor_model.optimizer.step()

        self.buffer.update_weights(targets)
        self.updates += 1
        if self.updates % 10 == 0:
            self.sync_nets()

        return loss.item()
