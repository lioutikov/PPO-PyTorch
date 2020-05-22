import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import os
from procgen import ProcgenEnv

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.discounted_returns = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.discounted_returns[:]

class EpisodicMemory:

    def __init__(self, num_parallel_episodes=1):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

        self._indices = np.empty(num_parallel_episodes,dtype=np.int)
        self._indices[:] = np.nan

        self._finished = np.zeros(num_parallel_episodes, dtype=bool)

    def reset(self, do_reset=None):
        if do_reset is None:
            do_reset = [True for _ in range(len(self._finished))]

        for i,d in enumerate(do_reset):
            if not d:
                continue
            if not self._finished[i]:
                self._finish(i)

            self._indices[i] = len(self.states)
            self.states.append([])
            self.actions.append([])
            self.logprobs.append([])
            self.rewards.append([])
            self._finished[i] = False

    def add(self, **kwargs):
        for i, finished in enumerate(self._finished):
            if not finished:
                for k,v in kwargs.items():
                    getattr(self,k)[self._indices[i]].append(v[i])

    def _finish(self, idx):
        self._finished[idx] = True

    def finish(self, did_finish):
        for i in range(len(self._finished)):
            if (not self._finished[i]) and did_finish[i]:
                self._finish(i)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

        self._indices[:] = np.nan
        self._finished[:] = False

    def avg_timesteps(self):
        return np.sum([len(states) for states in self.states])/len(self.states)

    def avg_return(self):
        return np.sum([np.sum(rewards) for rewards in self.rewards])/len(self.rewards)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self, state):
        return self.act(state)

    def acts(self, states, ememory=None):
        states = torch.from_numpy(states).float().to(device)
        action_probs = self.action_layer(states)
        dists = Categorical(action_probs)
        actions = dists.sample()
        logprobs = dists.log_prob(actions)

        if ememory is not None:
            ememory.add(states=states,actions=actions,logprobs=logprobs)

        return actions.detach().numpy(), logprobs.detach().numpy()

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob)

        return action.item(),logprob

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, ememory):

        discounted_returns = []
        for rewards in ememory.rewards:
            discounted_returns.append([])
            discounted_return = 0
            for reward in reversed(rewards):
                discounted_return = reward + (self.gamma * discounted_return)
                discounted_returns[-1].insert(0,discounted_return)


        # Normalizing the returns:

        discounted_returns = torch.cat([torch.Tensor(drets) for drets in discounted_returns]).to(device)#.detach()
        # discounted_returns = torch.tensor(discounted_returns).to(device)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

        old_states = torch.cat([torch.stack(states) for states in ememory.states]).to(device).detach()
        old_actions = torch.cat([torch.stack(actions) for actions in ememory.actions]).to(device).detach()
        old_logprobs = torch.cat([torch.stack(logprobs) for logprobs in ememory.logprobs]).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = discounted_returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, discounted_returns) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class DictToVec():

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, states_as_dict):
        return np.concatenate([states_as_dict[k] for k in self.keys],axis=1)

class StateTransformerEnv():

    def __init__(self, env, transform=None):
        self.env = env
        self.transform = transform

    @property
    def num_envs(self):
        return self.env.num_envs

    def reset(self):
        init_states = self.env.reset()
        if self.transform is not None:
            init_states = self.transform(init_states)
        return init_states

    def step(self, actions):
        next_states, rewards, dones, infos = self.env.step(actions)
        if self.transform is not None:
            next_states = self.transform(next_states)
        return next_states, rewards, dones, infos

    def render(self):
        return self.env.render()

def create_collector_env_gym(seed):

    options = {
        "world_dim": int(16),
        "init_locator_type": int(3),
        "num_goals_green": int(1),
        "num_goals_red": int(1),
        "num_resources_green": int(0),
        "num_resources_red": int(0),
        "num_fuel": int(0),
        "num_obstacles": int(2),
        "goal_max": 20.0,
        "goal_init": 0.0,
        "agent_max_fuel": 100.0,
        "agent_init_fuel": 100.0,
        "agent_max_resources": 100.0,
        "agent_init_resources_green": 20.0,
        "agent_init_resources_red": 10.0,
    }

    kwargs = {
        "start_level": seed if seed is not None else 102,
        "num_levels": 10,
        "additional_obs_spaces": [
            ProcgenEnv.C_Space("state_ship", False, (9,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_goals", False, ((options["num_goals_green"]+options["num_goals_red"])*4,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_resources", False, ((options["num_resources_green"]+options["num_resources_red"])*4,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_obstacles", False, (options["num_obstacles"]*3,), float, (-1e6,1e6))
        ],
        'max_episodes_per_game': 0,
        "options": options
        }

    # env = gym.make('procgen:procgen-collector-v0',**kwargs)

    env = ProcgenEnv(num_envs=9, env_name="collector", **kwargs)
    dtv = DictToVec([space.name for space in kwargs["additional_obs_spaces"] if np.prod(space.shape) > 0])
    env = StateTransformerEnv(env, dtv)
    return env


def main():
    ############## Hyperparameters ##############
    env_name = "procgen:procgen-collector-v0"
    # creating environment
    # env = gym.make(env_name)

    import sys

    render = True
    solved_reward = 20030         # stop training if avg_reward > solved_reward
    log_interval = 100           # print avg reward in the interval
    save_interval = 1000           # print avg reward in the interval
    max_episodes = 200000        # max training episodes
    max_timesteps = 2000         # max timesteps in one episode
    n_latent_var = 128           # number of variables in hidden layer
    # update_timestep = 1000      # update policy every n timesteps
    update_episodes = 10      # update policy every n episodes
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 10                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = int(sys.argv[1])
    #############################################

    env = create_collector_env_gym(random_seed)
    torch.manual_seed(random_seed)
    # state_dim = env.observation_space.shape[0]
    state_dim = 23
    action_dim = 9
    # print(env.observation_space,env.action_space)
    # if random_seed:
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)

    os.makedirs(f"seed_{random_seed}", exist_ok=True)

    ememory = EpisodicMemory(env.num_envs)

    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    from pprint import pprint

    # training loop
    for i_episode in range(1, max_episodes+1):
        states = env.reset()
        ememory.reset()
        if render and (i_episode%(log_interval) == 0):
            env.render()

        for t in range(max_timesteps):

            # Running policy_old:
            actions, logprobs = ppo.policy_old.acts(states,ememory)

            states, rewards, dones, _ = env.step(actions)

            ememory.add(rewards=rewards)
            ememory.finish(dones)

            if render and (i_episode%(log_interval) == 0):
                env.render()

            if dones.all():
                break




        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), f'seed_{random_seed}/PPO_{env_name}_solved.pt')
        #     break

        # logging
        if i_episode % log_interval == 0:
            print('Episode {} \t avg length: {} \t avg return: {}'.format(i_episode, ememory.avg_timesteps(), ememory.avg_return()))

        # update if its time
        if i_episode % update_episodes == 0:
            ppo.update(ememory)
            ememory.clear()

        if i_episode % save_interval == 0:
            print(f"########## Saved: {i_episode}##########")
            torch.save(ppo.policy.state_dict(), f'seed_{random_seed}/PPO_{env_name}_{i_episode:06d}.pt')
if __name__ == '__main__':
    main()
