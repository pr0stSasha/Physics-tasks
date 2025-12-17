import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

G = 9.81
MASS_DEFAULT = 1.0
K_QUADRATIC = 0.01
K_LINEAR = 0.5
V_THRESHOLD = 1e-8
V0_HIGH = 80
V0_LOW = 15
ANGLE_MAIN = 45
MAX_TIME = 200
INTERVAL_MS = 1
FRAMES_COUNT = 150
ANGLES_LIST = [30, 45, 60]
VELOCITIES_LIST = [40, 60, 80]
MASSES_HIGH_V = [1, 5, 20]
K_VALUES_LIST = [0.2, 0.5, 1.0]
MASSES_LOW_V = [0.5, 1, 2]

def simulate_projectile(v0, angle_deg, k, mode, mass=MASS_DEFAULT, g=G, max_time=MAX_TIME):
    angle_rad = np.radians(angle_deg)
    state0 = [0, 0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)]

    def derivatives(t, state):
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)
        if v < V_THRESHOLD:
            return [vx, vy, 0, -g]
        if mode == 'linear':
            F_res = k * v
        elif mode == 'quadratic':
            F_res = k * v**2
        else:
            F_res = 0
        dvx_dt = -F_res * vx / (v * mass)
        dvy_dt = -g - F_res * vy / (v * mass)
        return [vx, vy, dvx_dt, dvy_dt]

    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(derivatives, [0, max_time], state0, events=hit_ground, 
                    dense_output=True, rtol=1e-7)

    t_final = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
    t_eval = np.linspace(0, t_final, FRAMES_COUNT)
    traj = sol.sol(t_eval)
    return traj[0], traj[1]

def animate_session(session_name, experiments):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Сессия: {session_name}", fontsize=16, fontweight='bold')
    axs = axs.flatten()

    lines_storage, points_storage = [], []

    for i, (title, exp_list) in enumerate(experiments):
        ax = axs[i]
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

        all_x = [max(d[0]) for d, _, _ in exp_list]
        all_y = [max(d[1]) for d, _, _ in exp_list]
        ax.set_xlim(0, max(all_x) * 1.1)
        ax.set_ylim(0, max(all_y) * 1.2)
        
        curr_lines, curr_points = [], []
        for (x, y), label, color in exp_list:
            plot_color = color if color else None
            line, = ax.plot([], [], label=label, color=plot_color, lw=2)
            point, = ax.plot([], [], 'o', color=line.get_color(), markersize=5)
            curr_lines.append((x, y, line))
            curr_points.append(point)
        ax.legend(fontsize='small')
        lines_storage.append(curr_lines)
        points_storage.append(curr_points)

    def update(frame):
        artists = []
        for i in range(len(lines_storage)):
            for j in range(len(lines_storage[i])):
                x_data, y_data, line_obj = lines_storage[i][j]
                point_obj = points_storage[i][j]
                idx = min(frame, len(x_data) - 1)
                line_obj.set_data(x_data[:idx], y_data[:idx])
                point_obj.set_data([x_data[idx]], [y_data[idx]])
                artists.extend([line_obj, point_obj])
        return artists

    ani = FuncAnimation(fig, update, frames=FRAMES_COUNT, interval=INTERVAL_MS, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    session1 = [
        (f"Лобовое vs Вакуум (V0={V0_HIGH}, {ANGLE_MAIN}°)", [
            (simulate_projectile(V0_HIGH, ANGLE_MAIN, 0, 'none'), 'Вакуум', 'blue'),
            (simulate_projectile(V0_HIGH, ANGLE_MAIN, K_QUADRATIC, 'quadratic'), 'Лобовое', 'green')
        ]),
        (f"Углы (V0=70, Лобовое)", [
            (simulate_projectile(70, ang, K_QUADRATIC, 'quadratic'), f'Угол {ang}°', None) 
            for ang in ANGLES_LIST
        ]),
        (f"Разная V0 ({ANGLE_MAIN}°, Лобовое)", [
            (simulate_projectile(v, ANGLE_MAIN, K_QUADRATIC, 'quadratic'), f'V0={v}м/с', None) 
            for v in VELOCITIES_LIST
        ]),
        (f"Разная масса (V0=70, {ANGLE_MAIN}°)", [
            (simulate_projectile(70, ANGLE_MAIN, K_QUADRATIC, 'quadratic', mass=m_val), f'm={m_val}кг', None) 
            for m_val in MASSES_HIGH_V
        ])
    ]

    session2 = [
        (f"Вязкое vs Вакуум (V0={V0_LOW})", [
            (simulate_projectile(V0_LOW, ANGLE_MAIN, 0, 'none'), 'Вакуум', 'blue'),
            (simulate_projectile(V0_LOW, ANGLE_MAIN, K_LINEAR, 'linear'), 'Вязкое', 'orange')
        ]),
        ("Углы в вязкой среде", [
            (simulate_projectile(V0_LOW, ang, K_LINEAR, 'linear'), f'Угол {ang}°', None) 
            for ang in ANGLES_LIST
        ]),
        ("Разная вязкость k", [
            (simulate_projectile(V0_LOW, ANGLE_MAIN, k_val, 'linear'), f'k={k_val}', None) 
            for k_val in K_VALUES_LIST
        ]),
        ("Разная масса", [
            (simulate_projectile(V0_LOW, ANGLE_MAIN, K_LINEAR, 'linear', mass=m_val), f'm={m_val}кг', None) 
            for m_val in MASSES_LOW_V
        ])
    ]

    animate_session("Высокие скорости и лобовое сопротивление", session1)
    animate_session("Вязкая среда и малые скорости", session2)