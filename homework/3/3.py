import numpy as np


def bellman_backup(states, actions, rewards, transition_probs, gamma, threshold=1e-6):
    num_states = len(states)
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    while True:
        delta = 0
        for s in states:
            if not actions(s):
                continue

            action_values = []
            for a in actions(s):
                value = sum(
                    transition_probs[s][a][next_s] * (rewards[s][a][next_s] + gamma * value_function[next_s])
                    for next_s in states
                )
                action_values.append(value)

            best_value = max(action_values)
            delta = max(delta, abs(value_function[s] - best_value))
            value_function[s] = best_value
            policy[s] = actions(s)[np.argmax(action_values)]

        if delta < threshold:
            break

    return policy, value_function


def main():
    states = list(range(3))
    actions = lambda s: [0, 1]

    rewards = np.array([
        [[0, 1, -1], [0, 0, 1]],
        [[1, -1, 0], [-1, 1, 0]],
        [[0, 1, 0], [1, 0, -1]]
    ])

    transition_probs = np.array([
        [[0.8, 0.2, 0.0], [0.1, 0.6, 0.3]],
        [[0.0, 0.9, 0.1], [0.4, 0.5, 0.1]],
        [[0.3, 0.3, 0.4], [0.2, 0.3, 0.5]]
    ])

    gamma = 0.9

    policy, value_function = bellman_backup(states, actions, rewards, transition_probs, gamma)

    print("\nOptimal Policy:\n", policy)
    print("Value Function:\n", value_function)


if __name__ == "__main__":
    main()
