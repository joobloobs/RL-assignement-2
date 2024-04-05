import gym
from duelingDQN import dqn, DuelingDQNAgent
import matplotlib.pyplot as plt
from vars import *
from collections import namedtuple


ENVS = [HyperEnv('CartPole-v1', 1000, 1000, 128, 1e-4, 100, 0.99, 0.1, 128, 64), HyperEnv('Acrobot-v1', 200, 500, 256, 1e-3, 500, 0.99, 0.1, 128, 256)]
# Define main function
def main():

    for hyperEnv in ENVS:
        env = gym.make(hyperEnv.name)

        # Initialize agents
        dueling_dqn_agent_type1 = DuelingDQNAgent(env.observation_space.shape[0], env.action_space.n, 37, "mean", hyperEnv)
        dueling_dqn_agent_type2 = DuelingDQNAgent(env.observation_space.shape[0], env.action_space.n, 37, "max", hyperEnv)
        #monte_carlo_reinforce_agent = MonteCarloREINFORCEAgent(state_size, action_size)

        # Train and evaluate Dueling DQN agents
        print(f"\nTraining and evaluating Dueling DQN agents on {hyperEnv.name} environment:")
        print("Type 1:")
        scores = dqn(dueling_dqn_agent_type1, env, n_episodes=hyperEnv.n_episodes, max_t=hyperEnv.max_t)
        plt.plot(scores)
        plt.show()

        """
        env = gym.make(hyperEnv.name, render_mode="human")
        state = env.reset()[0]
        for t in range(hyperEnv.max_t):
            action = dueling_dqn_agent_type1.act(state)
            next_state, reward, done, _, _ = env.step(action)
            dueling_dqn_agent_type1.step(state, action, reward, next_state, done)
            state = next_state
            env.render()
            if done:
                break
"""
        print("Type 2:")
        # Train dueling DQN agent type 2
        scores = dqn(dueling_dqn_agent_type2, env, n_episodes=hyperEnv.n_episodes, max_t=hyperEnv.max_t)
        plt.plot(scores)
        plt.show()

        # Train and evaluate Monte Carlo REINFORCE agent
        print(f"\nTraining and evaluating Monte Carlo REINFORCE agent on {hyperEnv.name} environment:")
        # Train monte carlo REINFORCE agent


if __name__ == "__main__":
    main()
