import os
from math import exp

import matplotlib.pyplot as plt
import numpy as np


student_id = "2515070"
output_dir = "results"
depth_step = 10
pressure_steps = 100

if len(student_id) != 7 or not student_id.isdecimal():
    raise ValueError("Enter a seven-digit student ID.")
d, e, f, g = [int(digit) for digit in student_id[-4:]]
gas_gravity = 0.65 + (g + 1) / 50
h2s = (d + 1) / 100
co2 = (e + 1) / 100
n2 = (f + 1) / 100

p_wh = 100 + (e + 1) * 10
p_bh = 1000 + (e + 1) * 200
t_wh_f = 90 + (f + 1) * 5
t_bh_f = t_wh_f + 60 + (f + 1) * 5
well_depth = 2000 + (g + 1) * 100

r = 10.73
mw_air = 28.97
p_std = 14.7
t_std = 520.0


def critical_properties(gravity, h2s, co2, n2):
    hc = 1 - h2s - co2 - n2
    if min(h2s, co2, n2) < 0 or hc <= 0:
        raise ValueError("Check the gas mole fractions.")
    mw_impurities = 34.08 * h2s + 44.01 * co2 + 28.01 * n2
    gravity_hc = (gravity - mw_impurities / mw_air) / hc
    if gravity_hc <= 0:
        raise ValueError("Check the total gas gravity and mole fractions.")

    t_hc = 168 + 325 * gravity_hc - 12.5 * gravity_hc**2
    p_hc = 677 + 15 * gravity_hc - 37.5 * gravity_hc**2
    tpc = hc * t_hc + 672.4 * h2s + 547.9 * co2 + 227.5 * n2
    ppc = hc * p_hc + 1306 * h2s + 1071 * co2 + 493.1 * n2

    acid_gas = co2 + h2s
    epsilon = 120 * (acid_gas**0.9 - acid_gas**1.6) + 15 * (h2s**0.5 - h2s**4)
    tpc_corrected = tpc - epsilon
    ppc_corrected = ppc * tpc_corrected / (tpc + h2s * (1 - h2s) * epsilon)
    return tpc_corrected, ppc_corrected


def z_factor(p, t, tpc, ppc):
    pr = p / ppc
    tr = t / tpc
    a = 0.4278 * pr / tr**2.5
    b = 0.0867 * pr / tr
    roots = np.roots([1, -1, a - b - b**2, -a * b])
    return max(root.real for root in roots if abs(root.imag) < 1e-8 and root.real > b)


def gas_viscosity(t, rho, mw):
    k = 1e-4 * (9.4 + 0.02 * mw) * t**1.5 / (209 + 19 * mw + t)
    x = 3.5 + 986 / t + 0.01 * mw
    y = 2.4 - 0.2 * x
    return k * exp(x * (rho / 62.4)**y)


def ckb_viscosity(p, t_f, gravity, tpc, ppc, h2s, co2, n2):
    pr = p / ppc
    tr = (t_f + 459.67) / tpc
    if not (0.60 <= gravity <= 1.10 and 100 <= t_f <= 300
            and 1 <= pr <= 6 and 1.25 <= tr <= 2.20
            and 0 <= min(n2, co2, h2s) and max(n2, co2, h2s) <= 0.15):
        return np.nan

    gravities = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10]
    temperatures_f = [100, 200, 300]
    mu_1atm_chart = [
        [0.01114, 0.01072, 0.01036, 0.01005, 0.00976, 0.00949],
        [0.01277, 0.01230, 0.01192, 0.01156, 0.01124, 0.01096],
        [0.01440, 0.01393, 0.01352, 0.01314, 0.01280, 0.01246],
    ]
    mu_by_temperature = [np.interp(gravity, gravities, row) for row in mu_1atm_chart]
    mu_1atm = np.interp(t_f, temperatures_f, mu_by_temperature)

    correction_gravities = [0.60, 1.00, 1.50]
    delta_n2 = np.interp(gravity, correction_gravities, [0.00078, 0.00097, 0.00109])
    delta_co2 = np.interp(gravity, correction_gravities, [0.00048, 0.00068, 0.00080])
    delta_h2s = np.interp(gravity, correction_gravities, [0.00019, 0.00040, 0.00054])
    mu_1atm += (n2 * delta_n2 + co2 * delta_co2 + h2s * delta_h2s) / 0.10

    reduced_pressures = [1, 2, 3, 4, 6]
    reduced_temperatures = [1.25, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20]
    ratio_chart = [
        [1.113, 1.537, 2.212, 2.852, 3.797],
        [1.103, 1.440, 1.981, 2.520, 3.344],
        [1.088, 1.327, 1.678, 2.086, 2.707],
        [1.078, 1.256, 1.492, 1.791, 2.293],
        [1.067, 1.208, 1.385, 1.593, 2.020],
        [1.060, 1.177, 1.319, 1.478, 1.832],
        [1.053, 1.150, 1.268, 1.398, 1.689],
        [1.047, 1.132, 1.228, 1.340, 1.577],
        [1.043, 1.117, 1.200, 1.293, 1.492],
        [1.036, 1.099, 1.173, 1.256, 1.421],
        [1.033, 1.088, 1.154, 1.224, 1.367],
    ]
    ratios = [np.interp(pr, reduced_pressures, row) for row in ratio_chart]
    return mu_1atm * np.interp(tr, reduced_temperatures, ratios)


def pseudo_pressure(p, t, tpc, ppc, mw, steps=pressure_steps):
    pressures = np.linspace(p_std, p, steps + 1)
    values = []
    for pressure in pressures:
        z = z_factor(pressure, t, tpc, ppc)
        rho = mw * pressure / (z * r * t)
        mu = gas_viscosity(t, rho, mw)
        values.append(2 * pressure / (mu * z))

    dp = (p - p_std) / steps
    return dp * (sum(values[1:-1]) + (values[0] + values[-1]) / 2)


def main():
    tpc, ppc = critical_properties(gas_gravity, h2s, co2, n2)
    mw = mw_air * gas_gravity
    depths = list(range(0, well_depth, depth_step)) + [well_depth]
    rows = []

    for depth in depths:
        fraction = depth / well_depth
        p = p_wh + fraction * (p_bh - p_wh)
        t_f = t_wh_f + fraction * (t_bh_f - t_wh_f)
        t = t_f + 459.67
        z = z_factor(p, t, tpc, ppc)
        rho = mw * p / (z * r * t)
        mu = gas_viscosity(t, rho, mw)
        bg = (p_std / t_std) * z * t / p

        dp = 1.0
        z_low = z_factor(p - dp, t, tpc, ppc)
        z_high = z_factor(p + dp, t, tpc, ppc)
        dz_dp = (z_high - z_low) / (2 * dp)
        cg = 1 / p - dz_dp / z
        m = pseudo_pressure(p, t, tpc, ppc, mw)
        rows.append([depth, p, t_f, z, rho, mu, bg, cg, m])

    rows = np.array(rows)
    os.makedirs(output_dir, exist_ok=True)
    header = ("depth_ft,pressure_psia,temperature_F,z_factor,gas_density_lbm_ft3,"
              "gas_viscosity_cP,gas_fvf_rcf_scf,gas_compressibility_per_psia,pseudopressure_psi2_per_cP")
    np.savetxt(os.path.join(output_dir, "wellbore_properties.csv"), rows, delimiter=",",
               fmt="%.10g", header=header, comments="")

    profiles = [
        ("gas_density", "Gas Density", "Gas density (lbm/ft^3)", rows[:, 4]),
        ("gas_viscosity", "Gas Viscosity", "Gas viscosity (cP)", rows[:, 5]),
        ("gas_fvf", "Gas Formation Volume Factor", "Formation volume factor (rcf/scf)", rows[:, 6]),
        ("gas_compressibility", "Isothermal Gas Compressibility",
         "Gas compressibility (10^-3 /psia)", rows[:, 7] * 1000),
        ("pseudopressure", "Gas Pseudo-Pressure", "Pseudo-pressure (10^6 psi^2/cP)", rows[:, 8] / 1e6),
        ("z_factor", "Z-Factor", "Z-factor", rows[:, 3]),
    ]
    for filename, title, label, data in profiles:
        plt.figure(figsize=(8, 5))
        plt.plot(data, rows[:, 0], "o-", markersize=3, markevery=20)
        plt.gca().invert_yaxis()
        plt.title(title + " vs Depth")
        plt.xlabel(label)
        plt.ylabel("Depth (ft)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    pressures = np.linspace(p_std, 5000, 120)
    t_avg = (t_wh_f + t_bh_f) / 2
    temperatures = [t_avg - 100, t_avg - 50, t_avg, t_avg + 50, t_avg + 100]
    for temp_f in temperatures:
        values = [z_factor(p, temp_f + 459.67, tpc, ppc) for p in pressures]
        plt.plot(pressures, values, label=f"{temp_f:g} F")
    plt.plot(rows[:, 1], rows[:, 3], "k--", label="Well profile")
    plt.title("Z-Factor Isotherms")
    plt.xlabel("Pressure (psia)")
    plt.ylabel("Z-factor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "z_isotherms.png"), dpi=300)
    plt.close()

    gravities = np.linspace(gas_gravity - 0.03, gas_gravity + 0.17, 41)
    t = t_bh_f + 459.67
    mu_lee = []
    mu_ckb = []
    for gravity in gravities:
        tc, pc = critical_properties(gravity, h2s, co2, n2)
        z = z_factor(p_bh, t, tc, pc)
        mw = mw_air * gravity
        rho = mw * p_bh / (z * r * t)
        mu_lee.append(gas_viscosity(t, rho, mw))
        mu_ckb.append(ckb_viscosity(p_bh, t_bh_f, gravity, tc, pc, h2s, co2, n2))

    plt.figure(figsize=(8, 5))
    plt.plot(gravities, mu_lee, label="Lee-Gonzales-Eakin")
    plt.plot(gravities, mu_ckb, "--", label="Carr-Kobayashi-Burrows (charts)")
    plt.axvline(gas_gravity, color="black", linestyle=":", label=f"Base gravity {gas_gravity:.2f}")
    plt.title(f"Gas Viscosity vs Gas Gravity at {p_bh} psia and {t_bh_f} F")
    plt.xlabel("Total gas gravity (air = 1)")
    plt.ylabel("Gas viscosity (cP)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viscosity_sensitivity.png"), dpi=300)
    plt.close()

    print(f"Tpc = {tpc:.2f} R, Ppc = {ppc:.2f} psia")
    print(f"Saved {len(rows)} depth points and eight plots in {output_dir}.")


if __name__ == "__main__":
    main()
