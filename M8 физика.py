import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Константы
N = 256
dt = 0.004
rcut = 2.5
sigma = 1.0
epsilon = 1.0
kB = 1.0
m = 1.0

# Параметры процессов
rho_list = [0.3, 0.4, 0.5, 0.6]
T_list = [0.8, 1.0, 1.2]
P_targets = [0.6, 1.0, 2.0]

# Предвычисленные константы для потенциала с обрезанием
rc2 = rcut**2
inv_rc2 = 1.0 / rc2
inv_rc6 = inv_rc2**3
inv_rc12 = inv_rc6**2
U_rcut = 4*epsilon*(inv_rc12 - inv_rc6)

# Утилиты
def init_positions(N, L):
    """Кубическая решётка для стабильного старта"""
    n_side = int(np.ceil(N**(1/3)))
    grid = np.linspace(0, L, n_side, endpoint=False)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    pos = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]
    return pos

def init_velocities(N, T):
    """Максвелловское распределение с коррекцией"""
    v = np.random.randn(N,3)
    v -= v.mean(axis=0)  # убираем дрейф
    T_now = np.mean(np.sum(v**2,axis=1))*m/(3*kB)
    v *= math.sqrt(T/T_now)  # точная температура
    return v

def compute_forces_vectorized(pos, L):
    """Векторизованный расчёт сил Леннард-Джонса"""
    Nloc = pos.shape[0]
    rij = pos[:, None, :] - pos[None, :, :]
    rij -= L * np.round(rij / L)  # PBC
    r2 = np.sum(rij**2, axis=2)
    mask = (r2 > 1e-2) & (r2 < rc2)  # исключаем self и дальние
    i_idx, j_idx = np.nonzero(np.triu(mask,1))
    if len(i_idx)==0:
        return np.zeros_like(pos),0.0,0.0
    rij_pairs = rij[i_idx,j_idx,:]
    r2_pairs = r2[i_idx,j_idx]
    invr2 = 1.0 / r2_pairs
    invr6 = invr2**3
    invr12 = invr6**2
    U_pairs = 4*epsilon*(invr12 - invr6) - U_rcut  # потенциал
    U_total = np.sum(U_pairs)
    coef = 48*epsilon*(invr12 - 0.5*invr6)*invr2  # сила
    fij = coef[:, None] * rij_pairs
    forces = np.zeros((Nloc,3))
    np.add.at(forces, i_idx, fij)
    np.add.at(forces, j_idx, -fij)
    vir = np.sum(np.einsum('ij,ij->i', rij_pairs, fij))  # вириал
    return forces, U_total, vir

def vv_step(pos, vel, forces, L):
    """Один шаг Velocity Verlet"""
    pos_new = (pos + vel*dt + 0.5*forces/m*dt**2) % L
    f_new, U_new, vir_new = compute_forces_vectorized(pos_new,L)
    vel_new = vel + 0.5*(forces + f_new)/m*dt
    T_inst = np.mean(np.sum(vel_new**2,axis=1))*m/(3*kB)
    return pos_new, vel_new, f_new, U_new, vir_new, T_inst

def berendsen_rescale(vel,T_target,tau):
    """Простой термостат Берендсена"""
    T_now = np.mean(np.sum(vel**2,axis=1))*m/(3*kB)
    lam2 = 1 + (dt/tau)*(T_target/T_now - 1)
    if lam2>0:
        vel *= math.sqrt(lam2)
    return vel, T_now

def pressure_from_virial(T, vir, V):
    """Давление из вириальной теоремы"""
    rho = N/V
    return rho*kB*T + vir/(3*V)

def vdw_pressure(V, T, a, b):
    """Уравнение Ван-дер-Ваальса"""
    return (N*kB*T)/(V-N*b) - a*N**2/V**2

# NVT запуск
def run_NVT_once(rho, T_target, n_equil=100, n_prod=150, tau=2.0):
    """Один NVT запуск: эквилибровка + усреднение"""
    V = N/rho
    L = V**(1/3)
    pos = init_positions(N,L)
    vel = init_velocities(N,T_target)
    forces, U, vir = compute_forces_vectorized(pos,L)
    # эквилибровка
    for _ in range(n_equil):
        pos, vel, forces, U, vir, T_inst = vv_step(pos,vel,forces,L)
        vel,_ = berendsen_rescale(vel,T_target,tau)
    # продакшен
    P_samples=[]
    for _ in range(n_prod):
        pos, vel, forces, U, vir, T_inst = vv_step(pos,vel,forces,L)
        vel,_ = berendsen_rescale(vel,T_target,tau)
        P_samples.append(pressure_from_virial(T_inst,vir,V))
    return np.mean(P_samples), np.mean(U), pos, vel, forces, L

# Изобарический поиск V
def find_volume_for_pressure(P_target, T, V_low, V_high, tol=0.05, max_iter=8):
    """Бинарный поиск объёма для заданного давления"""
    for it in range(max_iter):
        V_mid = 0.5*(V_low+V_high)
        rho_mid = N/V_mid
        P_mid, _, _, _, _, _ = run_NVT_once(rho_mid,T,n_equil=50,n_prod=80)
        if abs(P_mid-P_target)<tol:
            return V_mid
        if P_mid>P_target:
            V_low=V_mid
        else:
            V_high=V_mid
    return 0.5*(V_low+V_high)

# Адиабат
def adiabatic_instant_jump(rho_init, factor, n_equil=100, n_nve=200):
    """Мгновенное сжатие + NVE динамика"""
    V0 = N/rho_init
    L0 = V0**(1/3)
    pos = init_positions(N,L0)
    vel = init_velocities(N,1.0)
    forces, U, vir = compute_forces_vectorized(pos,L0)
    # эквилибровка
    for _ in range(n_equil):
        pos, vel, forces, U, vir, T_inst = vv_step(pos, vel, forces, L0)
        vel,_ = berendsen_rescale(vel,1.0,2.0)
    # мгновенное сжатие
    V_new = V0*factor
    L_new = V_new**(1/3)
    pos = (pos * (L_new/L0)) % L_new
    forces, U, vir = compute_forces_vectorized(pos,L_new)
    temps=[]
    for _ in range(n_nve):
        pos, vel, forces, U, vir, T_inst = vv_step(pos,vel,forces,L_new)
        temps.append(T_inst)
    return np.array(temps), V_new

def main():
    t0=time.time()
    iso_data=[]
    
    # Изотермы
    for T in T_list:
        V_vals=[]
        P_vals=[]
        for rho in rho_list:
            P_mean,U_mean,_,_,_,_ = run_NVT_once(rho,T)
            V_vals.append(N/rho)
            P_vals.append(P_mean)
        iso_data.append((T,V_vals,P_vals))

    # параметры VdW (подобраны вручную)
    vdw_a=1.36
    vdw_b=0.063

    # Изохора
    V_fixed = N/0.6
    T_scan=[0.7,0.9,1.1,1.3]
    P_iso=[]
    for T in T_scan:
        P_mean, _, _, _, _, _ = run_NVT_once(N/V_fixed,T)
        P_iso.append(P_mean)

    # Изобара
    V_for_P=[]
    V_low = N/max(rho_list)
    V_high = N/min(rho_list)
    for P_target in P_targets:
        V_found = find_volume_for_pressure(P_target,1.0,V_low,V_high)
        V_for_P.append(V_found)

    # Адиабата
    comp_factors=[0.7,0.85,1.0,1.1,1.3]
    adiabatic_results=[]
    for f in comp_factors:
        temps, Vnew = adiabatic_instant_jump(0.6,f)
        adiabatic_results.append((f,np.mean(temps)))

    # Графики
    fig, axs = plt.subplots(2,2,figsize=(12,9))

    # Изотермы
    ax=axs[0,0]
    for T,V_vals,P_vals in iso_data:
        ax.plot(V_vals,P_vals,'o-',label=f"MD T={T}")
        Vrange=np.linspace(min(V_vals)*0.9,max(V_vals)*1.1,80)
        Pvdw=[vdw_pressure(V,T,vdw_a,vdw_b) for V in Vrange]
        ax.plot(Vrange,Pvdw,'--',alpha=0.7)
    ax.set_title("Изотермы: MD vs VdW")
    ax.set_xlabel("V")
    ax.set_ylabel("P")
    ax.grid(True)
    ax.legend()

    # Изохора
    ax=axs[0,1]
    ax.plot(T_scan,P_iso,'o-',label="MD")
    ax.plot(T_scan,[vdw_pressure(V_fixed,T,vdw_a,vdw_b) for T in T_scan],'x--',label="VdW")
    ax.set_title("Изохора")
    ax.set_xlabel("T")
    ax.set_ylabel("P")
    ax.grid(True)
    ax.legend()

    # Изобара
    ax=axs[1,0]
    ax.plot(P_targets,V_for_P,'o-')
    ax.set_title("Изобара (поиск V через NVT)")
    ax.set_xlabel("P_target")
    ax.set_ylabel("V_found")
    ax.grid(True)

    # Адиабата
    ax=axs[1,1]
    f_vals=[r[0] for r in adiabatic_results]
    Tvals=[r[1] for r in adiabatic_results]
    ax.plot(f_vals,Tvals,'o-')
    ax.set_title("Адиабатические прыжки")
    ax.set_xlabel("compression factor")
    ax.set_ylabel("T after NVE")
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    t1=time.time()
    print(f"Total time: {t1-t0:.1f}s")

if __name__=="__main__":
    main()