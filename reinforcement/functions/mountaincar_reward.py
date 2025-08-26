import numpy as np
import gymnasium as gym

# ------------------------------------------------------------------
# 1. 位置と速度からエネルギーを計算
# ------------------------------------------------------------------
def total_energy(state: np.ndarray, mass: float = 1.0, g: float = 9.81) -> float:
    """
    state: [position, velocity]
    Returns the sum of potential and kinetic energy.
    """
    pos, vel = state

    # ---- 位置エネルギー -----------------------------------------
    # MountainCar のトラック高さ: h(p) = sin(3p)*0.45 + 0.55
    height = np.sin(3 * pos) * 0.45 + 0.55
    potential = mass * g * height

    # ---- 運動エネルギー ---------------------------------------
    kinetic = 0.5 * mass * vel ** 2

    return potential + kinetic


# ------------------------------------------------------------------
# 2. 最大総エネルギー（最高地点で速度 0 のとき）を求める
# ------------------------------------------------------------------
def max_total_energy(mass: float = 1.0, g: float = 9.81,
                     n_samples: int = 5000) -> float:
    """
    位置の範囲 [-1.2, 0.6] を十分に細かくサンプリングし、  
    最高地点の高さで計算した位置エネルギーを返す。
    """
    # 位置をサンプリング
    pos_samples = np.linspace(-1.2, 0.6, n_samples)

    # 高さを計算
    heights = np.sin(3 * pos_samples) * 0.45 + 0.55
    h_max = heights.max()

    # 速度が 0.07 なので運動エネルギーは 0
    return mass * g * h_max + 0.5 * mass * 0.07 ** 2





MAX_E = max_total_energy()

print(MAX_E)
