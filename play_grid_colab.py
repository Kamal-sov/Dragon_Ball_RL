# === File: play_grid_colab.py ===
import torch
import time
from dragonball_env import DragonBallEnv
from dqn_agent import DQN

# Load the trained model
model = DQN()
model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Display helper
def print_grid(env):
    grid = [["." for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    x, y = env.agent_pos
    grid[y][x] = "G"
    for ex, ey in env.enemies:
        if grid[ey][ex] == ".":
            grid[ey][ex] = "E"
    for bx, by in env.beans:
        if grid[by][bx] == ".":
            grid[by][bx] = "S"  # Sensu bean
    for dx, dy in env.dragonballs:
        if grid[dy][dx] == ".":
            grid[dy][dx] = "⭐"

    print("\n".join([" ".join(row) for row in grid]))
    print(f"Form: {env.form} | HP: {env.hp:.1f} | EP: {env.ep} | Balls: {env.collected_balls}")
    print(f"Steps: {env.steps} | Status: {'DEAD' if env.hp <= 0 else 'ONGOING'}")
    print("-" * 30)

# Run one episode
env = DragonBallEnv()
state = env.reset()

done = False
print("\n--- Starting DQN Agent Game ---\n")

while not done:
    print_grid(env)
    time.sleep(0.2)  # For visibility

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_vals = model(state_tensor)
        action = torch.argmax(q_vals).item()

    state, reward, done = env.step(action)

print_grid(env)
print(f"\n✅ Game finished in {env.steps} steps!")

