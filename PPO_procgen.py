import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import os
from procgen import ProcgenEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
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
                nn.Linear(n_latent_var, 1)
                )

    def forward(self, state):
        return self.act(state)

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

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

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def create_collector_env_gym(seed):

    options = {
        "world_dim": int(16),
        "init_locator_type": int(3),
        "num_goals_green": int(1),
        "num_goals_red": int(1),
        "num_resources_green": int(0),
        "num_resources_red": int(0),
        "num_fuel": int(0),
        "num_obstacles": int(0),
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
        "num_levels": 1,
        "additional_obs_spaces": [
            ProcgenEnv.C_Space("state_ship", False, (9,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_goals", False, ((options["num_goals_green"]+options["num_goals_red"])*4,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_resources", False, ((options["num_resources_green"]+options["num_resources_red"])*4,), float, (-1e6,1e6)),
            ProcgenEnv.C_Space("state_obstacles", False, (options["num_obstacles"]*3,), float, (-1e6,1e6))
        ],
        'max_episodes_per_game': 0,
        "options": options
        }

    env = gym.make('procgen:procgen-collector-v0',**kwargs)

    return env


def main():
    ############## Hyperparameters ##############
    env_name = "procgen:procgen-collector-v0"
    # creating environment
    # env = gym.make(env_name)

    import sys

    render = True
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    save_interval = 1000           # print avg reward in the interval
    max_episodes = 100000        # max training episodes
    max_timesteps = 1000         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = int(sys.argv[1])
    #############################################

    env = create_collector_env_gym(random_seed)
    torch.manual_seed(random_seed)
    # state_dim = env.observation_space.shape[0]
    state_dim = 17
    action_dim = 9
    # print(env.observation_space,env.action_space)
    # if random_seed:
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)

    os.makedirs(f"seed_{random_seed}", exist_ok=True)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0


    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render and (i_episode%(log_interval*10) == 0):
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), f'seed_{random_seed}/PPO_{env_name}_solved.pt')
        #     break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        if i_episode % save_interval == 0:
            print(f"########## Saved: {i_episode}##########")
            torch.save(ppo.policy.state_dict(), f'seed_{random_seed}/PPO_{env_name}_{i_episode:06d}.pt')
if __name__ == '__main__':
    main()
