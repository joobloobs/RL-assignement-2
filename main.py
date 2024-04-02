import gym
from duelingDQN import DuelingDQNAgent

# Define a function to evaluate agent performance
def evaluate_agent(agent, env_name, episodes=100):
    pass


# Define main function
def main():
    env_names = ['Acrobot-v1', 'CartPole-v1']
    for env_name in env_names:
        env = gym.make(env_name)

        # Initialize agents
        dueling_dqn_agent_type1 = DuelingDQNAgent(env)
        #monte_carlo_reinforce_agent = MonteCarloREINFORCEAgent(state_size, action_size)

        # Train and evaluate Dueling DQN agents
        print(f"\nTraining and evaluating Dueling DQN agents on {env_name} environment:")
        print("Type 1:")
        # Train dueling DQN agent type 1
        print("Type 2:")
        # Train dueling DQN agent type 2

        # Train and evaluate Monte Carlo REINFORCE agent
        print(f"\nTraining and evaluating Monte Carlo REINFORCE agent on {env_name} environment:")
        # Train monte carlo REINFORCE agent


if __name__ == "__main__":
    main()
