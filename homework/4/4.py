import numpy as np
import matplotlib.pyplot as plt

# Define a simple grid environment
class SimpleGridEnv:
    def __init__(self):
        # Define a 4x4 grid
        self.grid = np.array([
            [0, 0, 0, 1],  # 1 means goal with reward 1
            [0, -1, 0, -1],  # -1 means obstacle
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.start = (3, 0)  # Starting point
        self.current_state = self.start
        self.actions = {
            0: (-1, 0),  # Move up
            1: (1, 0),   # Move down
            2: (0, -1),  # Move left
            3: (0, 1)    # Move right
        }

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        # Get the current position
        row, col = self.current_state
        move = self.actions[action]
        next_state = (row + move[0], col + move[1])

        # Check boundaries and obstacles
        if (0 <= next_state[0] < 4 and
            0 <= next_state[1] < 4 and
            self.grid[next_state] != -1):
            self.current_state = next_state
        else:
            next_state = self.current_state  # Invalid move, stay in place

        # Check if the agent reached the goal
        if self.grid[next_state] == 1:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty for each move
            done = False

        return next_state, reward, done

# Create the environment
env = SimpleGridEnv()

# Parameters
num_episodes = 1000
gamma = 0.99  # Discount factor
state_values = np.zeros((4, 4))  # Value for each state
returns = {state: [] for state in np.ndindex((4, 4))}  # Keep track of returns for each state

# Random policy: choose a random action
def policy(state):
    return np.random.choice(4)

# Monte Carlo First Visit Algorithm
for episode in range(num_episodes):
    state = env.reset()  # Start a new episode
    episode_data = []  # Store states, actions, and rewards

    done = False
    while not done:
        action = policy(state)  # Pick an action
        next_state, reward, done = env.step(action)  # Take the action
        episode_data.append((state, reward))  # Save the state and reward
        state = next_state

    # Calculate returns and update values
    G = 0
    visited_states = set()
    for state, reward in reversed(episode_data):
        G = gamma * G + reward  # Calculate return
        if state not in visited_states:
            visited_states.add(state)  # First visit
            returns[state].append(G)  # Save return
            state_values[state] = np.mean(returns[state])  # Update value

# Print the results
print("State Values:")
for row in state_values:
    print(["{:.2f}".format(v) for v in row])

# Visualize the state values
plt.imshow(state_values, cmap="coolwarm", origin="upper")
plt.colorbar(label="State Value")
plt.title("Monte Carlo: State Values")
plt.xticks(range(4))
plt.yticks(range(4))
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{state_values[i, j]:.2f}", ha="center", va="center", color="black")
plt.show()
