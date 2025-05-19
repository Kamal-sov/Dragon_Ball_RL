# === File: train_dqn_updated.py ===
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import matplotlib.pyplot as plt
from dqn_agent import DQN
from dragonball_env import DragonBallEnv

# Hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory = []
max_memory = 10000
train_episodes = 1000

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def remember(s, a, r, ns, d):
    if len(memory) > max_memory:
        memory.pop(0)
    memory.append((s, a, r, ns, d))

def replay():
    if len(memory) < batch_size:
        return
    samples = random.sample(memory, batch_size)
    states, actions_, rewards, next_states, dones = zip(*samples)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions_ = torch.tensor(actions_)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states)
    target_q = q_values.clone().detach()
    with torch.no_grad():
        next_q = model(next_states).max(1)[0]

    for i in range(batch_size):
        target_q[i][actions_[i]] = rewards[i] + gamma * next_q[i] * (1 - dones[i])

    output = model(states)
    loss = loss_fn(output, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
print("Training DQN Agent...")
rewards_plot = []

for episode in range(train_episodes):
    env = DragonBallEnv()
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = model(state_tensor)
                action = torch.argmax(q_vals).item()

        next_state, reward, done = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        replay()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_plot.append(total_reward)
    print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f}")

# Save model and rewards
torch.save(model.state_dict(), "dqn_model.pth")
with open("dqn_rewards.pkl", "wb") as f:
    pickle.dump(rewards_plot, f)

print("Training complete ✅ Model and rewards saved")

# === Plotting ===
# Optional: compute moving average for smoother graph
def moving_average(data, window=20):
    return [sum(data[i-window:i])/window if i >= window else sum(data[:i+1])/(i+1) for i in range(len(data))]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(rewards_plot, label="Episode Reward", alpha=0.5)
plt.plot(moving_average(rewards_plot), label="Moving Avg (20 episodes)", color='orange')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress of DQN Agent")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_rewards_graph.png")  # Saves image file for report
plt.show()
