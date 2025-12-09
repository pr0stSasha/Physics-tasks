import numpy as np
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# ======================================
#   НАСТРОЙКИ
# ======================================
N = 256          # сколько частиц
dt = 0.004       # шаг по времени
rcut = 2.5       # как далеко считать взаимодействие
sigma = 1.0      # размер частицы
epsilon = 1.0    # энергия взаимодействия
kB = 1.0         # константа Больцмана
m = 1.0          # масса частицы

# Что будем проверять
rho_list = [0.3, 0.4, 0.5, 0.6]    # разные плотности
T_list = [0.8, 1.0, 1.2]           # разные температуры
P_targets = [0.6, 1.0, 2.0]        # давления для изобары

# Константы для Леннард-Джонса
rc2 = rcut**2
inv_rc2 = 1.0 / rc2
inv_rc6 = inv_rc2**3
inv_rc12 = inv_rc6**2
U_rcut = 4*epsilon*(inv_rc12 - inv_rc6)  # сдвиг потенциала в ноль


# ======================================
#   НАЧАЛЬНЫЕ УСЛОВИЯ
# ======================================
def init_positions(N, L):
    # Ставим частицы на кубическую решетку
    n_side = int(np.ceil(N**(1/3)))  # сколько частиц по одной стороне
    grid = np.linspace(0, L, n_side, endpoint=False)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    pos = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]
    return pos


def init_velocities(N, T):
    # Случайные скорости
    v = np.random.randn(N, 3)
    v -= v.mean(axis=0)  # чтоб центр масс не двигался
    T_now = np.mean(np.sum(v**2, axis=1))*m/(3*kB)
    v *= math.sqrt(T/T_now)  # подгоняем под нужную температуру
    return v


# ======================================
#   СИЛЫ (Леннард-Джонс)
# ======================================
def compute_forces_vectorized(pos, L):
    # Разности координат
    Nloc = pos.shape[0]
    rij = pos[:, None, :] - pos[None, :, :]
    rij -= L * np.round(rij / L)  # периодические границы
    r2 = np.sum(rij**2, axis=2)

    # Ищем пары частиц, которые близко
    mask = (r2 > 1e-2) & (r2 < rc2)
    i_idx, j_idx = np.nonzero(np.triu(mask, 1))

    if len(i_idx) == 0:
        return np.zeros_like(pos), 0.0, 0.0

    # Берем только нужные пары
    rij_pairs = rij[i_idx, j_idx, :]
    r2_pairs = r2[i_idx, j_idx]

    # Формулы Леннард-Джонса
    invr2 = 1.0 / r2_pairs
    invr6 = invr2**3
    invr12 = invr6**2

    U_pairs = 4*epsilon*(invr12 - invr6) - U_rcut
    U_total = np.sum(U_pairs)

    # Сила = -dU/dr
    coef = 48*epsilon*(invr12 - 0.5*invr6) * invr2
    fij = coef[:, None] * rij_pairs

    # Распределяем силы по частицам
    forces = np.zeros((Nloc, 3))
    np.add.at(forces, i_idx, fij)   # сила на первую частицу
    np.add.at(forces, j_idx, -fij)  # реакция на вторую

    # Вириал для давления
    vir = np.sum(np.einsum("ij,ij->i", rij_pairs, fij))

    return forces, U_total, vir


# ======================================
#   ИНТЕГРАЦИЯ (Velocity Verlet)
# ======================================
def vv_step(pos, vel, forces, L):
    # Координаты
    pos_new = (pos + vel*dt + 0.5*forces/m*dt**2) % L
    
    # Новые силы
    f_new, U_new, vir_new = compute_forces_vectorized(pos_new, L)
    
    # Скорости
    vel_new = vel + 0.5*(forces + f_new)/m*dt
    
    # Текущая температура
    T_inst = np.mean(np.sum(vel_new**2, axis=1))*m/(3*kB)
    
    return pos_new, vel_new, f_new, U_new, vir_new, T_inst


# ======================================
#   ТЕРМОСТАТ (Берендсен)
# ======================================
def berendsen_rescale(vel, T_target, tau):
    # Просто умножаем скорости на коэфф
    T_now = np.mean(np.sum(vel**2, axis=1))*m/(3*kB)
    lam2 = 1 + (dt/tau)*(T_target/T_now - 1)
    if lam2 > 0:
        vel *= math.sqrt(lam2)
    return vel, T_now


# ======================================
#   ДАВЛЕНИЕ
# ======================================
def pressure_from_virial(T, vir, V):
    # Формула из вириальной теоремы
    rho = N/V
    return rho*kB*T + vir/(3*V)


def vdw_pressure(V, T, a, b):
    # Ван-дер-Ваальс
    return (N*kB*T)/(V-N*b) - a*N**2/V**2


# ======================================
#   NVT СИМУЛЯЦИЯ (основная)
# ======================================
def run_NVT_once(rho, T_target, n_equil=100, n_prod=150, tau=2.0):
    V = N/rho
    L = V**(1/3)

    # Начальное состояние
    pos = init_positions(N, L)
    vel = init_velocities(N, T_target)
    forces, U, vir = compute_forces_vectorized(pos, L)

    # Сначала эквилибровка
    for _ in range(n_equil):
        pos, vel, forces, U, vir, T_inst = vv_step(pos, vel, forces, L)
        vel, _ = berendsen_rescale(vel, T_target, tau)

    # Потом сбор данных
    P_samples = []
    for _ in range(n_prod):
        pos, vel, forces, U, vir, T_inst = vv_step(pos, vel, forces, L)
        vel, _ = berendsen_rescale(vel, T_target, tau)
        P_samples.append(pressure_from_virial(T_inst, vir, V))

    return np.mean(P_samples), np.mean(U), pos, vel, forces, L


# ======================================
#   ИЗОБАРА (ищем объем для давления)
# ======================================
def find_volume_for_pressure(P_target, T, V_low, V_high, tol=0.05, max_iter=8):
    # Простой бинарный поиск
    for it in range(max_iter):
        V_mid = 0.5*(V_low + V_high)
        rho_mid = N/V_mid
        P_mid, _, _, _, _, _ = run_NVT_once(rho_mid, T, n_equil=50, n_prod=80)

        if abs(P_mid - P_target) < tol:
            return V_mid
        if P_mid > P_target:
            V_low = V_mid
        else:
            V_high = V_mid

    return 0.5*(V_low+V_high)


# ======================================
#   АДИАБАТА (резко меняем объем)
# ======================================
def adiabatic_instant_jump(rho_init, factor, n_equil=100, n_nve=200):
    # Начальный объем
    V0 = N/rho_init
    L0 = V0**(1/3)

    # Эквалибровка
    pos = init_positions(N, L0)
    vel = init_velocities(N, 1.0)
    forces, U, vir = compute_forces_vectorized(pos, L0)

    for _ in range(n_equil):
        pos, vel, forces, U, vir, T = vv_step(pos, vel, forces, L0)
        vel, _ = berendsen_rescale(vel, 1.0, 2.0)

    # Резкое сжатие
    V_new = V0 * factor
    L_new = V_new**(1/3)
    pos = (pos * (L_new/L0)) % L_new
    forces, U, vir = compute_forces_vectorized(pos, L_new)

    # NVE динамика (без термостата)
    temps = []
    for _ in range(n_nve):
        pos, vel, forces, U, vir, T = vv_step(pos, vel, forces, L_new)
        temps.append(T)

    return np.array(temps), V_new


# ======================================
#   КАРТИНКИ
# ======================================
def visualize_positions(pos, L, title="Частицы"):
    # Просто точки в 3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, alpha=0.85)

    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.set_zlim([0, L])
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def animate_md(n_steps, pos, vel, forces, L, dt_vis=10):
    # Простая анимация
    for step in range(n_steps):
        pos, vel, forces, U, vir, T = vv_step(pos, vel, forces, L)

        if step % dt_vis == 0:
            plt.clf()
            ax = plt.axes(projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=12, alpha=0.8)

            ax.set_xlim([0, L])
            ax.set_ylim([0, L])
            ax.set_zlim([0, L])
            ax.set_box_aspect([1, 1, 1])

            ax.set_title(f"Шаг {step}")
            plt.pause(0.001)

    return pos, vel, forces


# ======================================
#   ГЛАВНАЯ
# ======================================
def main():
    t0 = time.time()
    iso_data = []

    # Изотермы
    for T in T_list:
        V_vals = []
        P_vals = []
        for rho in rho_list:
            P, U, pos, vel, forces, L = run_NVT_once(rho, T)
            V_vals.append(N/rho)
            P_vals.append(P)
        iso_data.append((T, V_vals, P_vals))

    # Параметры Ван-дер-Ваальса (просто взял)
    vdw_a = 1.36
    vdw_b = 0.063

    # Изохора
    V_fixed = N/0.6
    T_scan = [0.7, 0.9, 1.1, 1.3]
    P_iso = []
    for T in T_scan:
        P, _, _, _, _, _ = run_NVT_once(N/V_fixed, T)
        P_iso.append(P)

    # Изобара
    V_for_P = []
    V_low = N/max(rho_list)
    V_high = N/min(rho_list)
    for P_target in P_targets:
        V_found = find_volume_for_pressure(P_target, 1.0, V_low, V_high)
        V_for_P.append(V_found)

    # Адиабата
    comp_factors = [0.7, 0.85, 1.0, 1.1, 1.3]
    adiabatic_results = []
    for f in comp_factors:
        temps, Vnew = adiabatic_instant_jump(0.6, f)
        adiabatic_results.append((f, np.mean(temps)))

    # ===== ГРАФИКИ =====
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Изотермы
    ax = axs[0, 0]
    for T, V_vals, P_vals in iso_data:
        ax.plot(V_vals, P_vals, 'o-', label=f"MD T={T}")
        Vr = np.linspace(min(V_vals)*0.9, max(V_vals)*1.1, 80)
        Pvdw = [vdw_pressure(V, T, vdw_a, vdw_b) for V in Vr]
        ax.plot(Vr, Pvdw, '--', alpha=0.7)
    ax.set_title("Изотермы")
    ax.set_xlabel("Объем V")
    ax.set_ylabel("Давление P")
    ax.grid(True)
    ax.legend()

    # Изохора
    ax = axs[0, 1]
    ax.plot(T_scan, P_iso, 'o-', label="MD")
    ax.plot(T_scan, [vdw_pressure(V_fixed, T, vdw_a, vdw_b) for T in T_scan],
            'x--', label="VdW")
    ax.set_title("Изохора")
    ax.set_xlabel("Температура T")
    ax.set_ylabel("Давление P")
    ax.grid(True)
    ax.legend()

    # Изобара
    ax = axs[1, 0]
    ax.plot(P_targets, V_for_P, 'o-')
    ax.set_title("Изобара")
    ax.set_xlabel("Давление P")
    ax.set_ylabel("Объем V")
    ax.grid(True)

    # Адиабата
    ax = axs[1, 1]
    f_vals = [r[0] for r in adiabatic_results]
    T_vals = [r[1] for r in adiabatic_results]
    ax.plot(f_vals, T_vals, 'o-')
    ax.set_title("Адиабата")
    ax.set_xlabel("Во сколько раз сжали")
    ax.set_ylabel("Температура после")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Картинка с частицами
    print("Рисую частицы...")
    P, U, pos, vel, forces, L = run_NVT_once(0.5, 1.0)
    visualize_positions(pos, L, "Частицы газа")

    print("Анимирую...")
    animate_md(500, pos, vel, forces, L, dt_vis=10)

    print(f"Время работы: {time.time()-t0:.1f} сек")


if __name__ == "__main__":
    main()
