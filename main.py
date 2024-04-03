import gym
from duelingDQN import dqn, DuelingDQNAgent
import matplotlib.pyplot as plt


# Define main function
def main():
    env_names = ['Acrobot-v1', 'CartPole-v1']
    for env_name in env_names:
        env = gym.make(env_name)

        # Initialize agents
        dueling_dqn_agent_type1 = DuelingDQNAgent(env.observation_space.shape[0], env.action_space.n, seed=37)
        #monte_carlo_reinforce_agent = MonteCarloREINFORCEAgent(state_size, action_size)

        # Train and evaluate Dueling DQN agents
        print(f"\nTraining and evaluating Dueling DQN agents on {env_name} environment:")
        print("Type 1:")
        scores = dqn(dueling_dqn_agent_type1, env)
        plt.plot(scores)
        plt.show()

        env = gym.make(env_name, render_mode="human")
        state = env.reset()[0]
        for t in range(1000):
            action = dueling_dqn_agent_type1.act(state)
            next_state, reward, done, _, _ = env.step(action)
            dueling_dqn_agent_type1.step(state, action, reward, next_state, done)
            state = next_state
            env.render()
            if done:
                break

        print("Type 2:")
        # Train dueling DQN agent type 2

        # Train and evaluate Monte Carlo REINFORCE agent
        print(f"\nTraining and evaluating Monte Carlo REINFORCE agent on {env_name} environment:")
        # Train monte carlo REINFORCE agent


if __name__ == "__main__":
    main()
