import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import shutil

# تعریف پوشه‌ها برای ذخیره نمودارها و ویدیوها
plots_folder = "plots"
videos_folder = "videos"

# حذف پوشه‌های موجود و ایجاد پوشه‌های جدید
if os.path.exists(plots_folder):
    shutil.rmtree(plots_folder)
os.makedirs(plots_folder, exist_ok=True)

if os.path.exists(videos_folder):
    shutil.rmtree(videos_folder)
os.makedirs(videos_folder, exist_ok=True)

# ایجاد محیط FrozenLake
env_name = "FrozenLake-v1"
env = gym.make(env_name, is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# تابع انتخاب اکشن به روش epsilon-greedy
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

# اجرای الگوریتم Q-Learning
def run_q_learning(env, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9995, episodes=20000):
    Q = np.zeros((n_states, n_actions))
    rewards = []
    epsilon = epsilon_start
    for ep in range(episodes):
        state, info = env.reset(seed=ep)
        done = False
        ep_reward = 0
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    return Q, rewards

# اجرای الگوریتم Double Q-Learning
def run_double_q_learning(env, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9995, episodes=20000):
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    rewards = []
    epsilon = epsilon_start
    for ep in range(episodes):
        state, info = env.reset(seed=ep)
        done = False
        ep_reward = 0
        while not done:
            Q_sum = Q1 + Q2
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q_sum[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if np.random.rand() < 0.5:
                a_prime = np.argmax(Q1[next_state])
                Q1[state, action] += alpha * (reward + gamma * Q2[next_state, a_prime] - Q1[state, action])
            else:
                a_prime = np.argmax(Q2[next_state])
                Q2[state, action] += alpha * (reward + gamma * Q1[next_state, a_prime] - Q2[state, action])
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    return Q1, Q2, rewards

# تنظیم مقادیر پیش‌فرض برای پارامترها
alpha = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9995
episodes = 20000

# آموزش الگوریتم Q-Learning
print("Training Q-Learning...")
Q, q_rewards = run_q_learning(env, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, episodes)

# آموزش الگوریتم Double Q-Learning
print("Training Double Q-Learning...")
Q1, Q2, dq_rewards = run_double_q_learning(env, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, episodes)

# تعریف تابع میانگین متحرک
window = 1000
def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

q_ma = moving_average(q_rewards, window)
dq_ma = moving_average(dq_rewards, window)

# رسم نمودار مقایسه‌ای
plt.figure(figsize=(12, 8))
plt.plot(q_ma, label='Q-Learning')
plt.plot(dq_ma, label='Double Q-Learning')
plt.title('Convergence of Q-Learning and Double Q-Learning on FrozenLake')
plt.xlabel('Episode (moving avg over 1000 episodes)')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_folder, "convergence_plot_improved.png"))
plt.show()

# محاسبه نرخ موفقیت نهایی
final_success_q = sum(q_rewards[-1000:]) / 1000.0
final_success_dq = sum(dq_rewards[-1000:]) / 1000.0
print(f"Final success rate Q-Learning (last 1000 episodes): {final_success_q:.2f}")
print(f"Final success rate Double Q-Learning (last 1000 episodes): {final_success_dq:.2f}")

# تابع سیاست greedy
def greedy_policy(Q, state):
    return np.argmax(Q[state])

# ضبط ویدیوهای مربوط به Q-Learning
test_env_q = gym.make(env_name, is_slippery=True, render_mode='rgb_array')
test_env_q = RecordVideo(test_env_q, video_folder=videos_folder, name_prefix="frozenlake_q_learning")
num_test_episodes = 5
for ep in range(num_test_episodes):
    state, info = test_env_q.reset(seed=ep)
    done = False
    while not done:
        action = greedy_policy(Q, state)
        next_state, reward, terminated, truncated, info = test_env_q.step(action)
        done = terminated or truncated
        state = next_state
test_env_q.close()

# ضبط ویدیوهای مربوط به Double Q-Learning
def combined_Q(Q1, Q2, state):
    return np.argmax(Q1[state] + Q2[state])

test_env_dq = gym.make(env_name, is_slippery=True, render_mode='rgb_array')
test_env_dq = RecordVideo(test_env_dq, video_folder=videos_folder, name_prefix="frozenlake_double_q_learning")
for ep in range(num_test_episodes):
    state, info = test_env_dq.reset(seed=ep)
    done = False
    while not done:
        action = combined_Q(Q1, Q2, state)
        next_state, reward, terminated, truncated, info = test_env_dq.step(action)
        done = terminated or truncated
        state = next_state
test_env_dq.close()

print("Videos have been saved in the 'videos' folder.")
