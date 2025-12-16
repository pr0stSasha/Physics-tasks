import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


G = 9.81
DEFAULT_MASS = 1.0
MAX_TIME = 200
TIME_STEPS = 150
VELOCITY_EPS = 1e-8
SOLVER_RTOL = 1e-7


HIGH_SPEED_V0 = [40, 60, 80]
HIGH_SPEED_ANGLES = [30, 45, 60]
HIGH_SPEED_K = 0.01
HIGH_SPEED_MASSES = [1, 5, 20]

LOW_SPEED_V0 = 15
LOW_SPEED_ANGLES = [30, 45, 60]
LOW_SPEED_K_VALUES = [0.2, 0.5, 1.0]
LOW_SPEED_MASSES = [0.5, 1, 2]


def simulate_projectile(v0: float, angle_deg: float, k: float, mode: str,
                        mass: float = DEFAULT_MASS, g: float = G, max_time: float = MAX_TIME):
    
    angle_rad = np.radians(angle_deg)
    state0 = [0, 0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)]

    def derivatives(t, state):
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)
        if v < VELOCITY_EPS:
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

    sol = solve_ivp(
        derivatives,
        [0, max_time],
        state0,
        events=hit_ground,
        dense_output=True,
        rtol=SOLVER_RTOL
    )

    t_final = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
    t_eval = np.linspace(0, t_final, TIME_STEPS)
    traj = sol.sol(t_eval)

    return traj[0], traj[1]


def animate_session(session_name: str, experiments: list):

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

    ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=1, blit=True, repeat=False)
    plt.show()


if __name__ == "__main__":

    session1 = [
        ("Лобовое vs Вакуум (V0=80, 45°)", [
            (simulate_projectile(80, 45, 0, 'none'), 'Вакуум', 'blue'),
            (simulate_projectile(80, 45, HIGH_SPEED_K, 'quadratic'), 'Лобовое', 'green')
        ]),
        ("Углы (V0=70, Лобовое)", [
            (simulate_projectile(70, ang, HIGH_SPEED_K, 'quadratic'), f'Угол {ang}°', None)
            for ang in HIGH_SPEED_ANGLES
        ]),
        ("Разная V0 (45°, Лобовое)", [
            (simulate_projectile(v, 45, HIGH_SPEED_K, 'quadratic'), f'V0={v}м/с', None)
            for v in HIGH_SPEED_V0
        ]),
        ("Разная масса (V0=70, 45°)", [
            (simulate_projectile(70, 45, HIGH_SPEED_K, 'quadratic', mass=m_val), f'm={m_val}кг', None)
            for m_val in HIGH_SPEED_MASSES
        ])
    ]

    session2 = [
        ("Вязкое vs Вакуум (V0=15)", [
            (simulate_projectile(LOW_SPEED_V0, 45, 0, 'none'), 'Вакуум', 'blue'),
            (simulate_projectile(LOW_SPEED_V0, 45, 0.5, 'linear'), 'Вязкое', 'orange')
        ]),
        ("Углы в вязкой среде", [
            (simulate_projectile(LOW_SPEED_V0, ang, 0.5, 'linear'), f'Угол {ang}°', None)
            for ang in LOW_SPEED_ANGLES
        ]),
        ("Разная вязкость k", [
            (simulate_projectile(LOW_SPEED_V0, 45, k_val, 'linear'), f'k={k_val}', None)
            for k_val in LOW_SPEED_K_VALUES
        ]),
        ("Разная масса", [
            (simulate_projectile(LOW_SPEED_V0, 45, 0.5, 'linear', mass=m_val), f'm={m_val}кг', None)
            for m_val in LOW_SPEED_MASSES
        ])
    ]

    animate_session("Высокие скорости и лобовое сопротивление", session1)
    animate_session("Вязкая среда и малые скорости", session2)
