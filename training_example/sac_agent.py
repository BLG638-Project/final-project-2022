import torch
import itertools
from network import Actor, Critic
from utils import Memory
from torch.autograd import Variable


class Model(torch.nn.Module):
    def __init__(self, env, params, n_insize, n_outsize):
        super().__init__()
        self.n_states = n_insize
        self.n_actions = n_outsize
        self.gamma = params.gamma
        self.tau = params.tau
        self.alpha = params.alpha
        self.hidden_size = params.hidden

        # create actor and critic networks
        self.actor = Actor(self.n_states, self.hidden_size, self.n_actions)
        self.actor_target = Actor(self.n_states, self.hidden_size, self.n_actions)
        self.critic_1 = Critic(self.n_states + self.n_actions, self.hidden_size, self.n_actions)
        self.critic_target_1 = Critic(self.n_states + self.n_actions, self.hidden_size, self.n_actions)
        self.critic_2 = Critic(self.n_states + self.n_actions, self.hidden_size, self.n_actions)
        self.critic_target_2 = Critic(self.n_states + self.n_actions, self.hidden_size, self.n_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target_1.parameters():
            p.requires_grad = False
        for p in self.critic_target_2.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())

        # make a replay buffer memory ro store experiences
        self.memory = Memory(params.buffersize)

        # mse loss algoritm is applied
        self.critic_criterion = torch.nn.MSELoss()

        # define actor and critic network optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.lrpolicy)
        self.critic_optimizer = torch.optim.Adam(self.q_params, lr=params.lrvalue)

    # get actions
    def select_action(self, state, deterministic=True):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        with torch.no_grad():
            action, _ = self.actor.forward(state, deterministic, False)
        action = action.detach().cpu().numpy()[0]
        return action

    # compute q-loss
    def calculate_loss_q(self, data):
        states, actions, rewards, next_states, dones = data[0], data[1], data[2], data[3], data[4]

        q_1 = self.critic_1(states, actions)
        q_2 = self.critic_2(states, actions)

        with torch.no_grad():
            next_action, logp_next_action = self.actor(next_states)

            q_1_pi_target = self.critic_target_1(next_states, next_action)
            q_2_pi_target = self.critic_target_2(next_states, next_action)
            q_pi_target = torch.min(q_1_pi_target, q_2_pi_target)

            # apply q-function
            backup = rewards + self.gamma * (1 - dones) * (q_pi_target - self.alpha * logp_next_action)

        # get average q-loss from both critic networks
        loss_q_1 = ((q_1 - backup) ** 2).mean()
        loss_q_2 = ((q_2 - backup) ** 2).mean()
        loss_q = loss_q_1 + loss_q_2

        return loss_q

    # compute pi-loss
    def calculate_loss_pi(self, states):
        pi, logp_pi = self.actor(states)

        q_1_pi = self.critic_1(states, pi)
        q_2_pi = self.critic_2(states, pi)

        q_pi = torch.min(q_1_pi, q_2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi

    # update actor and critic networks
    def update(self, experience):
        states, actions, rewards, next_states, dones = experience

        # convert experience vectors to tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.tensor(dones, dtype=torch.uint8)

        # compute and backward q-loss for critic network
        self.critic_optimizer.zero_grad()
        loss_q = self.calculate_loss_q((states, actions, rewards, next_states, dones))
        loss_q.backward()
        self.critic_optimizer.step()

        # freezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = False

        # compute and backward q-loss for actor network
        self.actor_optimizer.zero_grad()
        loss_pi = self.calculate_loss_pi(states)
        loss_pi.backward()
        self.actor_optimizer.step()

        # unfreezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = True

        # update target networks
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def _totorch(self, container, dtype):
        if isinstance(container[0], torch.Tensor):
            tensor = torch.stack(container)
        else:
            tensor = torch.tensor(container, dtype=dtype)
        return tensor
