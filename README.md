# 🐉 DragonBallRL: Reinforcement Learning Agent in a Custom Grid World

## 🧠 Overview

**DragonBallRL** is a reinforcement learning project where an agent, inspired by *Goku* from the Dragon Ball universe, learns to navigate a 10x10 grid maze using Deep Q-Networks (DQN). The agent must intelligently collect **Dragon Balls**, consume **Sensu Beans** to regain health and energy, and strategically interact with **enemies** to survive and win.

The game simulates key elements of real-world decision-making such as risk management, resource collection, and long-term strategy — all learned from scratch via interaction with the environment.

---

## 🎮 How It Works

- **Grid World**: A 10x10 grid with randomly placed **Dragon Balls**, **Sensu Beans**, and **Enemies** at the start of each episode.
- **Agent Objective**: Collect 3 Dragon Balls while managing health and energy, using Sensu Beans wisely and avoiding unnecessary conflict.
- **Transformations**: Goku can transform into *Super Saiyan (SSJ)* or *Super Saiyan Blue (SSJB)* based on energy levels. Transformations reduce incoming damage but consume energy.

---

## 🧾 Agent Mechanics

- **State Representation**:  
  The agent's state includes its position, health, energy, transformation form, and relative distance to the nearest bean, enemy, and dragon ball.

- **Action Space**:  
  Four actions: Up, Down, Left, Right.

- **Reward System**:  
  - +10 for Sensu Bean  
  - +50 for Dragon Ball  
  - -10 to -5 for enemy hit (based on form)  
  - +200 for completing goal  
  - -50 for dying  
  - -0.1 per step (to encourage efficiency)

---

## 🤖 Learning Approach

- The agent is trained with a **Deep Q-Network (DQN)**.
- It uses an **ε-greedy exploration strategy**, with decay from 1.0 to 0.1 over 1000 episodes.
- A **replay buffer** of 10,000 steps is used to stabilize learning.
- The model is trained using **PyTorch**, and performance is tracked via total reward graphs and gameplay logs.

---

## 📊 Results

- After 1000 training episodes, the agent consistently completes missions in under 20 steps.
- The agent learns to:
  - Prioritize bean collection before combat
  - Use energy for timely transformations
  - Navigate efficiently toward objectives

A training graph shows rising episode rewards and a stable moving average, proving successful policy learning.

---

## 📌 Key Features

- Dynamic environment with randomized object placement
- Energy-based transformation system
- Combat-aware movement strategy
- Visual output of agent decisions and training progress

---

## ✅ Summary

This project demonstrates how a deep reinforcement learning agent can learn smart behaviors in a thematic, non-trivial environment using DQN. It highlights the importance of reward engineering, state design, and environmental diversity for building robust AI systems.

# Dragon_Ball_RL
