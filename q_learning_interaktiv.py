import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ============================================================
#  EINSTELLUNGEN
# ============================================================
ROWS, COLS = 6, 6
ACTIONS = 4
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # oben, unten, links, rechts
ACTION_ARROWS = [u'\u2191', u'\u2193', u'\u2190', u'\u2192']  # hoch runter links rechts

alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
episodes_total = 300

# ============================================================
#  ZUSTAND
# ============================================================
grid = np.zeros((ROWS, COLS), dtype=int)  # 0=frei, -1=Wand, 1=Ziel
start_pos = [0, 0]
goal_pos = [ROWS - 1, COLS - 1]
grid[goal_pos[0], goal_pos[1]] = 1

Q = np.zeros((ROWS * COLS, ACTIONS))
mode = ['wall']
training = [False]
agent_pos = [0, 0]
rewards = []
epsilon = [epsilon_start]

# ============================================================
#  FIGURE SETUP
# ============================================================
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1e1e2e')

ax_grid   = fig.add_axes([0.03, 0.18, 0.48, 0.75])
ax_reward = fig.add_axes([0.57, 0.55, 0.40, 0.38])
ax_q      = fig.add_axes([0.57, 0.18, 0.40, 0.30])

for ax in [ax_grid, ax_reward, ax_q]:
    ax.set_facecolor('#181825')

ax_btn_wall  = fig.add_axes([0.03, 0.04, 0.10, 0.07])
ax_btn_start = fig.add_axes([0.14, 0.04, 0.10, 0.07])
ax_btn_goal  = fig.add_axes([0.25, 0.04, 0.10, 0.07])
ax_btn_train = fig.add_axes([0.36, 0.04, 0.10, 0.07])
ax_btn_reset = fig.add_axes([0.47, 0.04, 0.10, 0.07])
ax_btn_run   = fig.add_axes([0.58, 0.04, 0.10, 0.07])

btn_style = dict(color='#313244', hovercolor='#45475a')
btn_wall  = Button(ax_btn_wall,  'Wand',  **btn_style)
btn_start = Button(ax_btn_start, 'Start', **btn_style)
btn_goal  = Button(ax_btn_goal,  'Ziel',  **btn_style)
btn_train = Button(ax_btn_train, 'Train', **btn_style)
btn_reset = Button(ax_btn_reset, 'Reset', **btn_style)
btn_run   = Button(ax_btn_run,   'Run',   **btn_style)

for btn in [btn_wall, btn_start, btn_goal, btn_train, btn_reset, btn_run]:
    btn.label.set_color('white')
    btn.label.set_fontsize(11)

COL_FREE  = '#313244'
COL_WALL  = '#11111b'
COL_GOAL  = '#a6e3a1'
COL_START = '#89b4fa'
COL_AGENT = '#f38ba8'
COL_EDGE  = '#45475a'

# ============================================================
#  GRID ZEICHNEN
# ============================================================
def draw_grid(show_agent=True, show_arrows=True):
    ax_grid.cla()
    ax_grid.set_facecolor('#181825')
    ax_grid.set_xlim(0, COLS)
    ax_grid.set_ylim(0, ROWS)
    ax_grid.set_aspect('equal')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title('Labyrinth  (klick zum Bearbeiten)', color='white', fontsize=12, pad=8)

    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r, c]
            is_start = (r == start_pos[0] and c == start_pos[1])
            is_goal  = (r == goal_pos[0]  and c == goal_pos[1])

            if val == -1:
                color = COL_WALL
            elif is_goal:
                color = COL_GOAL
            elif is_start:
                color = COL_START
            else:
                color = COL_FREE

            rect = plt.Rectangle((c, ROWS - 1 - r), 1, 1,
                                  linewidth=1.5, edgecolor=COL_EDGE,
                                  facecolor=color)
            ax_grid.add_patch(rect)

            is_agent = (r == agent_pos[0] and c == agent_pos[1])

            if show_agent and is_agent:
                circle = plt.Circle((c + 0.5, ROWS - 1 - r + 0.5), 0.28,
                                    color=COL_AGENT, zorder=5)
                ax_grid.add_patch(circle)
            elif show_arrows and val != -1 and not is_goal:
                state = r * COLS + c
                q_max = np.max(Q[state])
                if q_max > 0:
                    best = np.argmax(Q[state])
                    ax_grid.text(c + 0.5, ROWS - 1 - r + 0.5,
                                 ACTION_ARROWS[best],
                                 ha='center', va='center',
                                 fontsize=18, color='#cdd6f4', zorder=4)

            if is_goal:
                ax_grid.text(c + 0.5, ROWS - 1 - r + 0.5, 'ZIEL',
                             ha='center', va='center',
                             fontsize=9, color='#1e1e2e', fontweight='bold', zorder=4)
            elif is_start:
                ax_grid.text(c + 0.5, ROWS - 1 - r + 0.5, 'START',
                             ha='center', va='center',
                             fontsize=9, color='#1e1e2e', fontweight='bold', zorder=4)

    fig.canvas.draw_idle()

# ============================================================
#  REWARD PLOT
# ============================================================
def draw_reward():
    ax_reward.cla()
    ax_reward.set_facecolor('#181825')
    ax_reward.set_title('Reward pro Episode  (eps=' + str(round(epsilon[0], 2)) + ')',
                        color='white', fontsize=10)
    ax_reward.set_xlabel('Episode', color='#6c7086', fontsize=8)
    ax_reward.set_ylabel('Total Reward', color='#6c7086', fontsize=8)
    ax_reward.tick_params(colors='#6c7086')
    for spine in ax_reward.spines.values():
        spine.set_edgecolor('#45475a')
    if rewards:
        ax_reward.plot(rewards, color='#89b4fa', linewidth=1.2)
    fig.canvas.draw_idle()

# ============================================================
#  Q-TABELLE HEATMAP
# ============================================================
def draw_q():
    ax_q.cla()
    ax_q.set_facecolor('#181825')
    ax_q.set_title('Q-Tabelle (max Q pro Zustand)', color='white', fontsize=10)
    ax_q.tick_params(colors='#6c7086')
    for spine in ax_q.spines.values():
        spine.set_edgecolor('#45475a')
    q_max = np.max(Q, axis=1).reshape(ROWS, COLS)
    ax_q.imshow(q_max, cmap='Blues', aspect='auto')
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    fig.canvas.draw_idle()

# ============================================================
#  KLICK HANDLER
# ============================================================
def on_click(event):
    if event.inaxes != ax_grid or training[0]:
        return
    if event.xdata is None or event.ydata is None:
        return
    c = int(event.xdata)
    r = ROWS - 1 - int(event.ydata)
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return

    is_start = (r == start_pos[0] and c == start_pos[1])
    is_goal  = (r == goal_pos[0]  and c == goal_pos[1])

    if mode[0] == 'wall':
        if not is_start and not is_goal:
            grid[r, c] = -1 if grid[r, c] != -1 else 0

    elif mode[0] == 'start':
        if grid[r, c] != -1 and not is_goal:
            start_pos[0] = r
            start_pos[1] = c
            agent_pos[0] = r
            agent_pos[1] = c

    elif mode[0] == 'goal':
        if grid[r, c] != -1 and not is_start:
            grid[goal_pos[0], goal_pos[1]] = 0
            goal_pos[0] = r
            goal_pos[1] = c
            grid[r, c] = 1

    draw_grid(show_agent=True, show_arrows=False)

fig.canvas.mpl_connect('button_press_event', on_click)

# ============================================================
#  ENV STEP
# ============================================================
def step_env(r, c, action):
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or grid[nr, nc] == -1:
        return r, c, -0.5, False
    if nr == goal_pos[0] and nc == goal_pos[1]:
        return nr, nc, 1.0, True
    return nr, nc, -0.01, False

# ============================================================
#  BUTTON CALLBACKS
# ============================================================
def set_wall(e):
    mode[0] = 'wall'
    btn_wall.label.set_text('[Wand]')
    btn_start.label.set_text('Start')
    btn_goal.label.set_text('Ziel')

def set_start(e):
    mode[0] = 'start'
    btn_wall.label.set_text('Wand')
    btn_start.label.set_text('[Start]')
    btn_goal.label.set_text('Ziel')

def set_goal(e):
    mode[0] = 'goal'
    btn_wall.label.set_text('Wand')
    btn_start.label.set_text('Start')
    btn_goal.label.set_text('[Ziel]')

def reset(e):
    Q[:] = 0
    rewards.clear()
    epsilon[0] = epsilon_start
    agent_pos[0] = start_pos[0]
    agent_pos[1] = start_pos[1]
    training[0] = False
    draw_grid(show_agent=True, show_arrows=False)
    draw_reward()
    draw_q()

def train(e):
    if training[0]:
        return
    training[0] = True
    Q[:] = 0
    rewards.clear()
    epsilon[0] = epsilon_start

    for ep in range(episodes_total):
        r, c = start_pos[0], start_pos[1]
        total_reward = 0

        for _ in range(200):
            state = r * COLS + c
            if np.random.random() < epsilon[0]:
                action = np.random.randint(ACTIONS)
            else:
                action = np.argmax(Q[state])

            nr, nc, reward, done = step_env(r, c, action)
            new_state = nr * COLS + nc

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            r, c = nr, nc
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        epsilon[0] = max(0.01, epsilon[0] * 0.995)

        if ep % 30 == 0:
            agent_pos[0] = r
            agent_pos[1] = c
            draw_grid(show_agent=True, show_arrows=True)
            draw_reward()
            draw_q()
            plt.pause(0.001)

    training[0] = False
    agent_pos[0] = start_pos[0]
    agent_pos[1] = start_pos[1]
    draw_grid(show_agent=True, show_arrows=True)
    draw_reward()
    draw_q()
    print("Training fertig!")

def run_agent(e):
    if training[0]:
        return
    r, c = start_pos[0], start_pos[1]
    agent_pos[0] = r
    agent_pos[1] = c
    draw_grid(show_agent=True, show_arrows=True)
    plt.pause(0.3)

    for _ in range(100):
        state = r * COLS + c
        action = np.argmax(Q[state])
        nr, nc, reward, done = step_env(r, c, action)
        r, c = nr, nc
        agent_pos[0] = r
        agent_pos[1] = c
        draw_grid(show_agent=True, show_arrows=True)
        plt.pause(0.25)
        if done:
            print("Ziel erreicht!")
            break

btn_wall.on_clicked(set_wall)
btn_start.on_clicked(set_start)
btn_goal.on_clicked(set_goal)
btn_train.on_clicked(train)
btn_reset.on_clicked(reset)
btn_run.on_clicked(run_agent)

# ============================================================
#  START
# ============================================================
draw_grid(show_agent=True, show_arrows=False)
draw_reward()
draw_q()
plt.suptitle('Q-Learning  -  Interaktives Labyrinth', color='white',
             fontsize=14, fontweight='bold', y=0.98)
plt.show()