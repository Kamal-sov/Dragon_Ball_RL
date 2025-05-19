# === File: dragonball_env.py ===
import random

class DragonBallEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.max_steps = 300
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.hp = 3
        self.ep = 0
        self.form = "Base"
        self.form_steps = 0
        self.collected_balls = 0
        self.steps = 0
        self.done = False

        self.dragonballs = self._place_random(3, avoid=[tuple(self.agent_pos)])
        self.beans = self._place_random(5, avoid=self.dragonballs + [tuple(self.agent_pos)])
        self.enemies = self._place_random(3, avoid=self.dragonballs + self.beans + [tuple(self.agent_pos)])

        return self._get_state()

    def _place_random(self, count, avoid=[]):
        positions = set()
        while len(positions) < count:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in positions and pos not in avoid:
                positions.add(pos)
        return list(positions)

    def _get_state(self):
        def rel_pos(targets):
            if not targets:
                return [0, 0]
            closest = min(targets, key=lambda t: abs(self.agent_pos[0]-t[0]) + abs(self.agent_pos[1]-t[1]))
            return [closest[0] - self.agent_pos[0], closest[1] - self.agent_pos[1]]

        rel_bean = rel_pos(self.beans)
        rel_ball = rel_pos(self.dragonballs)
        rel_enemy = rel_pos(self.enemies)

        form_map = {"Base": 0.0, "SSJ": 0.5, "SSJB": 1.0}
        return [
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            rel_bean[0] / self.grid_size,
            rel_bean[1] / self.grid_size,
            rel_ball[0] / self.grid_size,
            rel_ball[1] / self.grid_size,
            rel_enemy[0] / self.grid_size,
            rel_enemy[1] / self.grid_size,
            self.hp / 3,
            self.ep / 5,
            form_map[self.form]
        ]

    def step(self, action_idx):
        if self.done:
            return self._get_state(), 0, True

        dx, dy = self.actions[action_idx]
        x, y = self.agent_pos
        nx, ny = max(0, min(x + dx, self.grid_size - 1)), max(0, min(y + dy, self.grid_size - 1))
        self.agent_pos = [nx, ny]

        pos = tuple(self.agent_pos)
        reward = -0.2  # exploration penalty
        self.steps += 1

        # Collect Sensu Bean
        if pos in self.beans:
            self.beans.remove(pos)
            self.hp = min(3, self.hp + 1)
            self.ep += 1
            reward = 10

        # Encounter Enemy
        if pos in self.enemies:
            if self.form == "Base":
                self.hp -= 1
                reward = -10
            elif self.form == "SSJ":
                self.hp -= 0.5
                reward = -5
            elif self.form == "SSJB":
                reward = 5  # successful hit
            if self.hp <= 0:
                self.done = True
                return self._get_state(), -50, True

        # Collect Dragon Ball
        if pos in self.dragonballs:
            self.dragonballs.remove(pos)
            self.collected_balls += 1
            reward = 50
            if self.collected_balls == 3:
                self.done = True
                return self._get_state(), 200, True

        # Power-ups
        if self.form == "Base" and self.ep >= 3:
            self.form = "SSJ"
            self.ep -= 3
            self.form_steps = 5
        elif self.form == "SSJ" and self.ep >= 3:
            self.form = "SSJB"
            self.ep -= 3
            self.form_steps = 3
        elif self.form in ["SSJ", "SSJB"]:
            self.form_steps -= 1
            if self.form_steps <= 0:
                self.form = "Base"

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done
