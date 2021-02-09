import torch
import torch.nn.functional as F


class Agent():

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

        q_values = torch.gather(self.model(states), dim=-1,
                                index=actions).squeeze().to(self.model.device)

        target_q_values = rewards + \
            (1 - dones) * self.model.discount * \
            self.model(next_states).max(dim=-1)[0].detach()

        targets = F.mse_loss(q_values, target_q_values,
                             reduction='none').detach().numpy()
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        self.model.optimizer.step()

        self.buffer.update_weights(targets)

        return loss.item()
