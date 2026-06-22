# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:16:21 2026

@author: KHOURYK
"""

"""
Age-Structured SEIR Measles Model
===================================
Features:
  - Age structure: 4 groups (0–4, 5–14, 15–44, 45+)
  - SEIR compartments per age group (S, E, I, R)
  - Vital dynamics: births replenish S[0], natural deaths deplete all compartments
  - Disease-induced mortality (CFR-derived)
  - Seasonality: sinusoidal forcing on beta (school-term driven)
  - POLYMOD-derived contact matrix (Mossong et al. 2008, approximated for general population)
  - Two-dose vaccination: V1 (single dose, partial protection), V2 (two dose, high protection)
  - Time unit: years, months, or days

Compartments per age group a:
    S[a]  - Susceptible
    E[a]  - Exposed (latent, not yet infectious)
    I[a]  - Infectious
    R[a]  - Recovered/immune
    V1[a] - Vaccinated (1 dose, ~93% efficacy)
    V2[a] - Vaccinated (2 doses, ~97% efficacy)
    D[a]  - Cumulative disease deaths

Differential Equations (per age group a):
    lambda[a] = sum_b( beta(t) * C[a,b] * I[b] / N[b] )   <- force of infection
    dS/dt  = births_into_a  - lambda[a]*S[a] - nu1[a]*S[a]  - mu*S[a]
    dE/dt  = lambda[a]*S[a] + lambda[a]*sigma1*V1[a] + lambda[a]*sigma2*V2[a]
             - delta*E[a] - mu*E[a]
    dI/dt  = delta*E[a] - gamma*I[a] - mu_d*I[a] - mu*I[a]
    dR/dt  = gamma*I[a] - mu*R[a]
    dV1/dt = nu1[a]*S[a] - nu2[a]*V1[a] - mu*V1[a]
    dV2/dt = nu2[a]*V1[a] - mu*V2[a]
    dD/dt  = mu_d * I[a]

Parameters:
    N_total       - initial total population
    age_fracs     - fraction of population in each age group
    beta0         - baseline transmission rate (derived from R0)
    alpha_season  - seasonal forcing amplitude (0 = none, 0.2 = strong)
    delta         - rate of leaving latent period (1/incubation_days * 365)
    gamma         - recovery rate (1/infectious_days * 365)
    mu            - natural death/birth rate per year
    CFR           - case fatality rate (fraction)
    nu1           - first-dose vaccination rate per age group per year
    nu2           - second-dose vaccination rate per age group per year
    sigma1        - vaccine-modified susceptibility, dose 1 (1 - efficacy1)
    sigma2        - vaccine-modified susceptibility, dose 2 (1 - efficacy2)
    t_duration    - simulation length
    t_unit        - "years", "months", or "days"
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════
# INPUT PARAMETERS  ← edit these
# ═══════════════════════════════════════════════════════════════

N_total    = 12_000   # Initial total population
t_duration = 120      # Duration of simulation
t_unit     = "days"   # "years", "months", or "days"

# --- Age groups ---
# Four groups: 0–4, 5–14, 15–44, 45+
AGE_LABELS  = ["0–4 yrs", "5–14 yrs", "15–44 yrs", "45+ yrs"]
N_AGES      = 4

# Approximate population fractions (general mixed-income country)
age_fracs = np.array([0.08, 0.15, 0.45, 0.32])
assert abs(age_fracs.sum() - 1.0) < 1e-9, "age_fracs must sum to 1"

# --- Disease parameters ---
incubation_days  = 12    # Measles latent period (10–14 days)
infectious_days  = 9     # Measles infectious period
delta = (1 / incubation_days) * 365   # Rate of leaving E → I  (/yr)
gamma = (1 / infectious_days)  * 365  # Recovery rate (/yr)

CFR   = 0.02    # Case fatality rate (2% — moderate resource setting)
mu_d  = (CFR / (1.0 - CFR)) * gamma   # Disease death rate (/yr)

# --- Baseline transmission ---
# R0 for measles ~ 12–18; we use 15.
# With age structure, beta0 is a scalar scaled by the contact matrix.
R0_target = 12

# --- Seasonality ---
# beta(t) = beta0 * (1 + alpha * cos(2*pi*t))
# alpha=0   → no seasonality
# alpha=0.2 → 20% amplitude (school-term forcing, realistic for measles)
alpha_season = .2

# --- Vital dynamics ---
# mu = crude birth rate = crude death rate (steady-state assumption)
# Global average ~18–20/1000/yr; use 0.018 for moderate-income setting
mu = 0.018   # Natural birth/death rate per year

# --- Initial conditions ---
# Fraction initially infected (seeded into age group 1: school-age)
I0_count   = 5     # Total initial infected, placed in age group 1 (5–14 yrs)
seed_age   = 0      # Age group index to seed infection (0-indexed)

# Prior immunity fractions per age group (from historical exposure/vaccination)
# Older age groups more likely to be immune from prior infection
prior_immune_frac = np.array([0.05, 0.20, 0.40, 0.5])

# --- Vaccination rates (/yr per susceptible) ---
# nu1[a]: rate at which S in age group a receive dose 1
# nu2[a]: rate at which V1 in age group a receive dose 2
# Primarily targeted at young children (age group 0: 0–4 yrs)
nu1 = np.array([0.01, 0.01, 0.02, 0.01])   # Dose 1 rates
nu2 = np.array([0, 0.08, 0.01, 0.005])  # Dose 2 rates

# Vaccine efficacy
efficacy1 = 0.93   # Single dose
efficacy2 = 0.97   # Two doses
sigma1 = 1 - efficacy1   # Residual susceptibility with 1 dose
sigma2 = 1 - efficacy2   # Residual susceptibility with 2 doses

# ═══════════════════════════════════════════════════════════════
# CONTACT MATRIX  (POLYMOD-derived, Mossong et al. 2008)
# Rows = age of individual, Cols = age of contact
# Approximate symmetric matrix for groups: 0–4, 5–14, 15–44, 45+
# Units: mean contacts per day (converted to /yr below)
# ═══════════════════════════════════════════════════════════════
C_daily = np.array([
    #  0–4   5–14  15–44  45+
    [  8.5,  1.0,   3.5,  0.5],   # 0–4 contacts
    [  3.0,  8.0,   3.0,  0.5],   # 5–14 contacts  (high within-school)
    [  6.5,  7.5,   7.0,  1.5],   # 15–44 contacts (high within-adult)
    [  3.5,  5.5,   2.0,  3.0],   # 45+ contacts
])
C = C_daily * 365   # Convert to per-year contacts

# ═══════════════════════════════════════════════════════════════
# TIME UNIT & RESOLUTION
# ═══════════════════════════════════════════════════════════════
t_unit = t_unit.strip().lower()
if t_unit not in ("years", "months", "days"):
    raise ValueError("t_unit must be 'years', 'months', or 'days'")

if t_unit == "years":
    t_years = float(t_duration)
    display_scale = 1
elif t_unit == "months":
    t_years = t_duration / 12
    display_scale = 12
elif t_unit == "days":
    t_years = t_duration / 365
    display_scale = 365

x_label = f"Time ({t_unit})"

# 4 time points per day for numerical stability with fast measles dynamics
n_steps    = max(int(t_years * 365 * 4), 500)
t_internal = np.linspace(0, t_years, n_steps)
t_display  = t_internal * display_scale

# ═══════════════════════════════════════════════════════════════
# INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════
N_age = N_total * age_fracs   # Population per age group

# R (prior immune), then distribute remainder to S, V
R0_age  = N_age * prior_immune_frac
S0_age  = N_age * (1 - prior_immune_frac)
E0_age  = np.zeros(N_AGES)
I0_age  = np.zeros(N_AGES)
V10_age = np.zeros(N_AGES)
V20_age = np.zeros(N_AGES)
D0_age  = np.zeros(N_AGES)

# Seed infection
I0_age[seed_age]  = min(I0_count, S0_age[seed_age])
S0_age[seed_age] -= I0_age[seed_age]

# State vector order: S, E, I, R, V1, V2, D  × N_AGES
def pack(S, E, I, R, V1, V2, D):
    return np.concatenate([S, E, I, R, V1, V2, D])

def unpack(y):
    n = N_AGES
    return (y[0*n:1*n], y[1*n:2*n], y[2*n:3*n],
            y[3*n:4*n], y[4*n:5*n], y[5*n:6*n], y[6*n:7*n])

y0 = pack(S0_age, E0_age, I0_age, R0_age, V10_age, V20_age, D0_age)

# ═══════════════════════════════════════════════════════════════
# DERIVE beta0 FROM R0
# ═══════════════════════════════════════════════════════════════
def compute_R0(beta0_val):
    """Compute R0 from NGM using initial susceptible fractions."""
    s_frac = S0_age / N_age
    exit_E  = delta + mu
    exit_I  = gamma + mu_d + mu
    NGM = beta0_val * np.diag(s_frac) @ (C / 365) / exit_I
    return np.max(np.real(np.linalg.eigvals(NGM)))

# Bisect to find beta0
lo, hi = 0.001, 2000.0
for _ in range(60):
    mid = (lo + hi) / 2
    if compute_R0(mid) < R0_target:
        lo = mid
    else:
        hi = mid
beta0 = (lo + hi) / 2

R0_check = compute_R0(beta0)

# ═══════════════════════════════════════════════════════════════
# PRINT PARAMETERS
# ═══════════════════════════════════════════════════════════════
print("=" * 58)
print("  Age-Structured SEIR Measles Model")
print("=" * 58)
print(f"  Total population (N)         : {N_total:,}")
print(f"  Age groups                   : {', '.join(AGE_LABELS)}")
print(f"  Age group sizes              : {', '.join(f'{int(n):,}' for n in N_age)}")
print(f"  Incubation period            : {incubation_days} days  → δ={delta:.2f}/yr")
print(f"  Infectious period            : {infectious_days} days  → γ={gamma:.2f}/yr")
print(f"  R0 target                    : {R0_target}")
print(f"  R0 (NGM check)               : {R0_check:.3f}")
print(f"  beta0                        : {beta0:.4f}/yr")
print(f"  Seasonal amplitude (alpha)   : {alpha_season}")
print(f"  Natural birth/death rate     : {mu:.4f}/yr ({mu*1000:.1f}/1000/yr)")
print(f"  Case fatality rate           : {CFR*100:.1f}%")
print(f"  Vaccine efficacy (1/2 dose)  : {efficacy1*100:.0f}% / {efficacy2*100:.0f}%")
print(f"  Simulation duration          : {t_duration} {t_unit}")
print("=" * 58)

# ═══════════════════════════════════════════════════════════════
# ODE SYSTEM
# ═══════════════════════════════════════════════════════════════
def model(y, t, beta0, alpha, delta, gamma, mu_d, mu, C, nu1, nu2, sigma1, sigma2, N_total):
    S, E, I, R, V1, V2, D = unpack(y)

    # Seasonal beta
    beta_t = beta0 * (1 + alpha * np.cos(2 * np.pi * t))

    # Current total living population per age group
    N_live = S + E + I + R + V1 + V2

    # Force of infection per age group
    with np.errstate(divide='ignore', invalid='ignore'):
        I_over_N = np.where(N_live > 0, I / N_live, 0.0)
    lam = beta_t * (C @ I_over_N)

    # Births: all enter S[0] (age group 0–4)
    birth_rate = mu * N_live.sum()
    births     = np.zeros(N_AGES)
    births[0]  = birth_rate

    dS  = births - lam * S  - nu1 * S  - mu * S
    dE  = lam * S + lam * sigma1 * V1 + lam * sigma2 * V2 - delta * E - mu * E
    dI  = delta * E - gamma * I - mu_d * I - mu * I
    dR  = gamma * I - mu * R
    dV1 = nu1 * S  - nu2 * V1 - mu * V1
    dV2 = nu2 * V1 - mu * V2
    dD  = mu_d * I

    return pack(dS, dE, dI, dR, dV1, dV2, dD)

# ═══════════════════════════════════════════════════════════════
# SOLVE
# ═══════════════════════════════════════════════════════════════
print("\n  Solving ODE ... ", end="", flush=True)
solution = odeint(
    model, y0, t_internal,
    args=(beta0, alpha_season, delta, gamma, mu_d, mu, C, nu1, nu2, sigma1, sigma2, N_total),
    mxstep=5000, rtol=1e-6, atol=1e-8
)
print("done.")

S_sol, E_sol, I_sol, R_sol, V1_sol, V2_sol, D_sol = unpack(solution.T)

# Aggregates across all age groups
S_tot  = S_sol.sum(axis=0)
E_tot  = E_sol.sum(axis=0)
I_tot  = I_sol.sum(axis=0)
R_tot  = R_sol.sum(axis=0)
V1_tot = V1_sol.sum(axis=0)
V2_tot = V2_sol.sum(axis=0)
D_tot  = D_sol.sum(axis=0)
N_live_tot = S_tot + E_tot + I_tot + R_tot + V1_tot + V2_tot

# Rt over time
def compute_Rt_series():
    Rt_arr = np.zeros(n_steps)
    exit_I = gamma + mu_d + mu
    for k in range(n_steps):
        s_frac_k = np.where(
            N_live_tot[k] > 0,
            S_sol[:, k] / np.maximum(S_sol[:, k] + E_sol[:, k] + I_sol[:, k] + R_sol[:, k] + V1_sol[:, k] + V2_sol[:, k], 1.0),
            0.0
        )
        beta_t = beta0 * (1 + alpha_season * np.cos(2 * np.pi * t_internal[k]))
        NGM_k  = beta_t * np.diag(s_frac_k) @ (C / 365) / exit_I
        Rt_arr[k] = max(np.max(np.real(np.linalg.eigvals(NGM_k))), 0.0)
    return Rt_arr

print("  Computing Rt series ... ", end="", flush=True)
Rt = compute_Rt_series()
print("done.\n")

# Summary stats
peak_I_total  = I_tot.max()
peak_t_disp   = t_display[np.argmax(I_tot)]
total_deaths  = D_tot[-1]
total_vacc    = (V1_tot + V2_tot)[-1]

print(f"  Peak infected (all ages)     : {peak_I_total:,.0f} ({peak_I_total/N_total*100:.1f}%)")
print(f"  Peak occurs at               : {peak_t_disp:.2f} {t_unit}")
print(f"  Total disease deaths         : {total_deaths:,.0f} ({total_deaths/N_total*100:.2f}%)")
print(f"  Total vaccinated (end)       : {total_vacc:,.0f} ({total_vacc/N_total*100:.1f}%)")
print(f"  Final population (living)    : {N_live_tot[-1]:,.0f}")

# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════
AGE_COLORS = ["#E8624C", "#F5A623", "#4C9BE8", "#4CE87A"]

fig = plt.figure(figsize=(14, 18))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.32)

fig.suptitle(
    f"Age-Structured SEIR Measles Model  |  N={N_total:,}  |  R₀={R0_target}  |  "
    f"α={alpha_season}  |  CFR={CFR*100:.0f}%  |  {t_duration} {t_unit}",
    fontsize=13, fontweight="bold", y=0.98
)

td = t_display

# ── Panel 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(td, S_tot,  color="#4C9BE8", lw=2, label="Susceptible (S)")
ax1.plot(td, E_tot,  color="#F5A623", lw=2, label="Exposed (E)")
ax1.plot(td, I_tot,  color="#E8624C", lw=2, label="Infectious (I)")
ax1.plot(td, R_tot,  color="#4CE87A", lw=2, label="Recovered (R)")
ax1.plot(td, N_live_tot, color="#AAAAAA", lw=1.2, ls=":", label="Living N")
ax1.set_title("Total Population — SEIR Compartments", fontsize=10)
ax1.set_ylabel("Individuals")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax1.legend(fontsize=8, loc="center right")
ax1.grid(True, alpha=0.3)

# ── Panel 2
ax2 = fig.add_subplot(gs[0, 1])
for a in range(N_AGES):
    ax2.plot(td, I_sol[a], color=AGE_COLORS[a], lw=2, label=AGE_LABELS[a])
ax2.set_title("Infectious Individuals by Age Group", fontsize=10)
ax2.set_ylabel("Infectious individuals")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Panel 3
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(td, V1_tot,       color="#8B5CF6", lw=2, label="Vaccinated dose 1 (V1)")
ax3.plot(td, V2_tot,       color="#5C3D8B", lw=2, label="Vaccinated dose 2 (V2)")
ax3.plot(td, D_tot,        color="#333333", lw=2, ls="--", label=f"Cumul. deaths (final: {total_deaths:,.0f})")
ax3.fill_between(td, V1_tot + V2_tot, alpha=0.08, color="#8B5CF6")
ax3.fill_between(td, D_tot,           alpha=0.12, color="#333333")
ax3.set_title("Vaccination Coverage & Cumulative Deaths", fontsize=10)
ax3.set_ylabel("Individuals")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Panel 4
ax4 = fig.add_subplot(gs[1, 1])
for a in range(N_AGES):
    ax4.plot(td, S_sol[a] / N_age[a] * 100, color=AGE_COLORS[a], lw=2, label=AGE_LABELS[a])
ax4.set_title("Susceptible Fraction by Age Group (%)", fontsize=10)
ax4.set_ylabel("% of age group susceptible")
ax4.set_ylim(0, 100)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Panel 5
ax5 = fig.add_subplot(gs[2, 0])
ax5.fill_between(td, Rt, 1, where=(Rt >= 1), interpolate=True,
                 color="#E8624C", alpha=0.15, label="Growing (Rt > 1)")
ax5.fill_between(td, Rt, 1, where=(Rt  < 1), interpolate=True,
                 color="#4CE87A", alpha=0.15, label="Declining (Rt < 1)")
ax5.plot(td, Rt, color="#8B5CF6", lw=2, label="Rt")
ax5.axhline(1.0, color="black", ls="--", lw=1.2, alpha=0.7, label="Rt = 1")
ax5.set_title("Effective Reproductive Number Rt Over Time", fontsize=10)
ax5.set_ylabel("Rt")
ax5.set_ylim(0, max(Rt.max() * 1.15, 1.5))
ax5.legend(fontsize=8, loc="upper right")
ax5.grid(True, alpha=0.3)

# ── Panel 6
ax6 = fig.add_subplot(gs[2, 1])
beta_series = beta0 * (1 + alpha_season * np.cos(2 * np.pi * t_internal))
ax6.plot(td, beta_series, color="#E8624C", lw=2, label="β(t) seasonal")
ax6.axhline(beta0, color="gray", ls="--", lw=1.2, label=f"β₀ = {beta0:.1f}")
ax6.set_title("Seasonal Forcing on Transmission Rate β(t)", fontsize=10)
ax6.set_ylabel("Transmission rate β(t) (/yr)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# ── Panel 7
ax7 = fig.add_subplot(gs[3, :])
dt_yr   = t_internal[1] - t_internal[0]
incid_total = delta * E_tot * dt_yr * 365
window = max(int(30 / 365 / dt_yr), 1)
incid_smooth = np.convolve(incid_total, np.ones(window)/window, mode='same')

ax7.fill_between(td, incid_smooth, alpha=0.3, color="#E8624C")
ax7.plot(td, incid_smooth, color="#E8624C", lw=1.5, label="Incidence (30-day smoothed)")
for a in range(N_AGES):
    incid_a = delta * E_sol[a] * dt_yr * 365
    incid_a_smooth = np.convolve(incid_a, np.ones(window)/window, mode='same')
    ax7.plot(td, incid_a_smooth, color=AGE_COLORS[a], lw=1.2,
             alpha=0.8, ls="--", label=AGE_LABELS[a])
ax7.set_title("Annualised Incidence Rate Over Time (New Infections/Year, 30-day smoothed)", fontsize=10)
ax7.set_ylabel("New infections / year")
ax7.set_xlabel(x_label, fontsize=11)
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax7.legend(fontsize=8, ncol=3)
ax7.grid(True, alpha=0.3)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlabel(x_label, fontsize=9)

plt.show()