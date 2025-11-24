import numpy as np
import os
import gymnasium as gym
import random

env = gym.make('Taxi-v3')

alpha = 0.9  # percentage of how important new information is
gamma = 0.95  # how important future rewards are
epsilon = 1.0  # randomness/exploration rate
epsilon_decay = 0.9995  # used to make epsilon smaller overtime
min_epsilon = 0.01  # min epsilon to ensure dont decay past 0
num_episodes = 10000  # how many times the game is ran
max_steps = 100

# 5x5 grid -> 25 positions * 5 * 4 = 500 different states 5 states 4 actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))  # initlizing qtable with 0 as we have no info


# each actionb has its own q value action is our y

def choose_action(state):  # chooses action based on random or best q value
    if random.uniform(0, 1) < epsilon:  # if random value is less than epsilon do a random action
        return env.action_space.sample()  # does any possible action
    else:
        return np.argmax(q_table[state, :])  # does best action according to Q-table


for episode in range(num_episodes):
    state = env.reset()[0]  # Get just the state from reset

    done = False

    for step in range(max_steps):
        action = choose_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]  # q value in table
        next_max = np.max(q_table[next_state, :])  # maximum Q value for that action

        q_table[state, action] = (1 - alpha) * old_value + alpha * (
                    reward + gamma * next_max)  # updates state for action based on the math

        state = next_state

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress every 1000 episodes
    if episode % 1000 == 0:
        print(f"Episode {episode}, Epsilon: {epsilon:.4f}")

# end of training process

print("Training completed!")
print(f"Final epsilon: {epsilon:.4f}")

env.close()

# Test the trained agent
env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state = env.reset()[0]
    done = False
    truncated = False

    print(f'Episode {episode}')

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])  # using knowledge we have to fill stable
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print(f'Finished episode {episode} with reward {reward}')
            print("hi")
            print(q_table)
            break




env.close()

