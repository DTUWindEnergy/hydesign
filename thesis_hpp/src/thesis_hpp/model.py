import pyomo.environ as pyo


VALID_SETTLEMENT_PERIODS = {"hourly", "daily", "weekly", "monthly", "yearly"}


def _build_settlement_periods(t_values, settlement_period):
    """Return ordered settlement-period hour groups for the 8,736-hour horizon."""
    t_values = list(t_values)
    settlement_period = str(settlement_period).strip().lower()

    if settlement_period not in VALID_SETTLEMENT_PERIODS:
        valid = ", ".join(sorted(VALID_SETTLEMENT_PERIODS))
        raise ValueError(
            f"Unknown settlement_period '{settlement_period}'. Choose one of: {valid}."
        )

    if settlement_period == "hourly":
        return {sp: [t] for sp, t in enumerate(t_values, start=1)}

    if settlement_period == "daily":
        if len(t_values) % 24 != 0:
            raise ValueError("Daily settlement requires a whole number of 24-hour days.")
        return {
            sp: t_values[start:start + 24]
            for sp, start in enumerate(range(0, len(t_values), 24), start=1)
        }

    if settlement_period == "weekly":
        hours_per_week = 24 * 7
        if len(t_values) % hours_per_week != 0:
            raise ValueError("Weekly settlement requires a whole number of 168-hour weeks.") 

        return {
            sp: t_values[start:start + hours_per_week]
            for sp, start in enumerate(
                range(0, len(t_values), hours_per_week),
                start=1,
            )
        }   

    if settlement_period == "yearly":
        return {1: t_values}

    # BTC uses the common synthetic calendar 2 January to 30 December 2012.
    # These month lengths sum to 364 days = 8,736 hours.
    if len(t_values) != 8_736:
        raise ValueError("Monthly settlement currently requires exactly 8,736 hours.")
    month_days = [30, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30]
    periods = {}
    start = 0
    for sp, days in enumerate(month_days, start=1):
        end = start + 24 * days
        periods[sp] = t_values[start:end]
        start = end
    return periods


def build_model(data):
    cfg = data["config"]

    settlement_period = cfg.get("settlement_period", "hourly")
    settlement_period_hours = _build_settlement_periods(
        data["T"], settlement_period
    )

    m = pyo.ConcreteModel(name="three_stage_hpp")

    # -------------------------
    # Sets
    # -------------------------
    m.T = pyo.Set(initialize=data["T"], ordered=True)
    m.Omega = pyo.Set(initialize=data["Omega"], ordered=True)
    m.Xi = pyo.Set(initialize=data["Xi"], ordered=True)
    m.SP = pyo.Set(initialize=list(settlement_period_hours), ordered=True)
    m.T_in_SP = pyo.Set(
        m.SP,
        initialize=lambda _m, sp: settlement_period_hours[sp],
        ordered=True,
    )

    # Plain Python metadata used when exporting results.
    m._settlement_period_name = str(settlement_period).strip().lower()

    # -------------------------
    # Parameters
    # -------------------------
    m.lambda_spot = pyo.Param(m.T, m.Omega, initialize=data["lambda_spot"])
    m.lambda_mfrr_up = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["lambda_mfrr_up"])
    m.lambda_mfrr_down = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["lambda_mfrr_down"])
    m.lambda_imb = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["lambda_imb"])
    m.y_up = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["y_up"])
    m.y_down = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["y_down"])

    m.sdp = pyo.Param(m.T, initialize=data["sdp"])
    m.pi_omega = pyo.Param(m.Omega, initialize=data["pi_omega"])
    m.pi_xi_given_omega = pyo.Param(m.Omega, m.Xi, initialize=data["pi_xi_given_omega"])

    # Kept as input even if unused right now
    m.wind_forecast = pyo.Param(m.T, m.Omega, initialize=data["wind_forecast"])
    m.wind_real = pyo.Param(m.T, m.Omega, m.Xi, initialize=data["wind_real"])

    # Scalar parameters
    m.lambda_PPA = pyo.Param(initialize=cfg["lambda_PPA"])
    m.c_imb = pyo.Param(initialize=cfg["c_imb"])
    m.c_ppa = pyo.Param(initialize=cfg["c_ppa"])
    m.degradation_cost = pyo.Param(initialize=cfg["degradation_cost_per_mw"])
    m.delta_t = pyo.Param(initialize=cfg.get("delta_t_hours", 1.0))

    m.P_nom = pyo.Param(initialize=cfg["P_nom"])
    m.P_b_max = pyo.Param(initialize=cfg["P_b_max"])
    m.P_grid = pyo.Param(initialize=cfg["P_grid"])

    m.eta_dis = pyo.Param(initialize=cfg["eta_dis"])
    m.eta_cha = pyo.Param(initialize=cfg["eta_cha"])
    m.E_max = pyo.Param(initialize=cfg["E_max"])
    m.SoC_min = pyo.Param(initialize=cfg["SoC_min"])
    m.SoC_max = pyo.Param(initialize=cfg["SoC_max"])
    m.E0 = pyo.Param(initialize=cfg["E0"])

    m.alpha = pyo.Param(initialize=cfg["alpha"])
    m.beta = pyo.Param(initialize=cfg["beta"])

    m.A_up_max = pyo.Param(m.T, m.Omega, initialize=data["A_up_max"])
    m.A_down_max = pyo.Param(m.T, m.Omega, initialize=data["A_down_max"])

    # -------------------------
    # Variables
    # -------------------------
    # Stage 1
    m.S_PPA = pyo.Var(domain=pyo.NonNegativeReals)

    # Derived first-stage PPA commitment
    m.P_PPA = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Stage 2
    m.P_DA = pyo.Var(m.T, m.Omega, domain=pyo.NonNegativeReals)

    # Stage 3
    m.a_up = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.a_down = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    m.P_cha_down = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals) # for modeling imbalance as separate charge/discharge
    m.curt_down = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals) # for modeling imbalance as separate curtailment
    m.P_cha_free = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals) # for modeling charging as free action plus charging from down-regulation
    m.curt_free = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals) # for modeling curtailment as free action plus curtailment from down-regulation

    m.S_short = pyo.Var(m.SP, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.curt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    m.P_dis_rt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.P_cha_rt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.P_PPA_rt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.P_rt_grid = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    m.delta = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.Reals)

    #m.z_rt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.Binary)
    m.E_rt = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    # Absolute value linearization for |delta|
    m.delta_pos = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)
    m.delta_neg = pyo.Var(m.T, m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    # Scenario profit
    m.Phi = pyo.Var(m.Omega, m.Xi, domain=pyo.Reals)

    # CVaR variables
    m.eta_cvar = pyo.Var(domain=pyo.Reals)
    m.u = pyo.Var(m.Omega, m.Xi, domain=pyo.NonNegativeReals)

    # -------------------------
    # Helper
    # -------------------------
    T_list = list(m.T)
    if not T_list:
        raise ValueError("The optimization horizon must contain at least one hour.")
    next_t = {
        T_list[idx]: T_list[idx + 1]
        for idx in range(len(T_list) - 1)
    }
    last_t = T_list[-1]
    first_hour_each_day = cfg.get("first_hour_each_day", [1])

    # -------------------------
    # Constraints
    # -------------------------

    # PPA commitment: P_t^PPA = sdp_t * S_PPA
    def ppa_commitment_rule(m, t):
        return m.P_PPA[t] == m.sdp[t] * m.S_PPA
    m.ppa_commitment = pyo.Constraint(m.T, rule=ppa_commitment_rule)

    # Stage-2 upper bound
    # def stage2_forecast_bound_rule(m, t, w):
    #     return m.P_DA[t, w] + m.P_PPA[t] <= m.wind_forecast[t, w] + m.P_b_max

    # m.stage2_forecast_bound = pyo.Constraint(m.T, m.Omega, rule=stage2_forecast_bound_rule)

    def stage2_grid_bound_rule(m, t, w):
        return m.P_DA[t, w] + m.P_PPA[t] <= m.P_grid

    m.stage2_grid_bound = pyo.Constraint(m.T, m.Omega, rule=stage2_grid_bound_rule)


    # Physical power balance
    def power_balance_rule(m, t, w, x):
        return m.P_rt_grid[t, w, x] == (
            m.wind_real[t, w, x] - m.curt[t, w, x]
            + m.P_dis_rt[t, w, x] - m.P_cha_rt[t, w, x]
        )
    m.power_balance = pyo.Constraint(m.T, m.Omega, m.Xi, rule=power_balance_rule)

    # Imbalance definition
    def imbalance_rule(m, t, w, x):
        return m.delta[t, w, x] == (
            m.P_rt_grid[t, w, x]
            - (m.P_DA[t, w] + m.P_PPA_rt[t, w, x] + m.a_up[t, w, x] - m.a_down[t, w, x])
        )
    m.imbalance_def = pyo.Constraint(m.T, m.Omega, m.Xi, rule=imbalance_rule)

    # Absolute value linearization
    def delta_split_rule(m, t, w, x):
        return m.delta[t, w, x] == m.delta_pos[t, w, x] - m.delta_neg[t, w, x]
    m.delta_split = pyo.Constraint(m.T, m.Omega, m.Xi, rule=delta_split_rule)

    def delta_abs_limit_rule(m, t, w, x):
        return m.delta_pos[t, w, x] + m.delta_neg[t, w, x] <= 0.05 * m.P_grid
    m.delta_abs_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=delta_abs_limit_rule)


    # Charging cannot come from curtailment
    def charging_from_down_rule(m, t, w, x):
        return m.P_cha_down[t, w, x] <= 0.0
    m.charging_from_down = pyo.Constraint(m.T, m.Omega, m.Xi, rule=charging_from_down_rule)

    # Net PPA shortfall over each settlement period.
    def ppa_shortfall_rule(m, sp, w, x):
        return m.S_short[sp, w, x] >= sum(
            (m.P_PPA[t] - m.P_PPA_rt[t, w, x]) * m.delta_t
            for t in m.T_in_SP[sp]
        )
    m.ppa_shortfall = pyo.Constraint(
        m.SP, m.Omega, m.Xi, rule=ppa_shortfall_rule
    )

    # Overdelivery may offset underdelivery within a settlement period, but
    # total credited PPA delivery cannot exceed contracted PPA energy.
    def ppa_period_delivery_cap_rule(m, sp, w, x):
        return sum(
            m.P_PPA_rt[t, w, x] * m.delta_t for t in m.T_in_SP[sp]
        ) <= sum(
            m.P_PPA[t] * m.delta_t for t in m.T_in_SP[sp]
        )
    m.ppa_period_delivery_cap = pyo.Constraint(
        m.SP, m.Omega, m.Xi, rule=ppa_period_delivery_cap_rule
    )

    # PPA delivery cannot exceed real-time grid injection
    def ppa_delivery_limit_rule(m, t, w, x):
        return m.P_PPA_rt[t, w, x] <= m.P_rt_grid[t, w, x]
    m.ppa_delivery_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=ppa_delivery_limit_rule)

    # Battery charge/discharge logic
    # def batt_dis_rt_rule(m, t, w, x):
    #     return m.P_dis_rt[t, w, x] <= m.P_b_max * m.z_rt[t, w, x]
    # m.batt_dis_rt = pyo.Constraint(m.T, m.Omega, m.Xi, rule=batt_dis_rt_rule)

    # def batt_cha_rt_rule(m, t, w, x):
    #     return m.P_cha_rt[t, w, x] <= m.P_b_max * (1 - m.z_rt[t, w, x])
    # m.batt_cha_rt = pyo.Constraint(m.T, m.Omega, m.Xi, rule=batt_cha_rt_rule)

    # Relaxed version without battery binary variable
    def batt_power_limit_rule_relaxed(m, t, w, x):
        return m.P_dis_rt[t, w, x] + m.P_cha_rt[t, w, x] <= m.P_b_max
    m.batt_power_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=batt_power_limit_rule_relaxed)

    # Real-time SoC evolution
    def soc_rt_rule(m, t, w, x):
        end_of_hour_soc = (
            m.E_rt[t, w, x]
            - m.P_dis_rt[t, w, x] / m.eta_dis
            + m.P_cha_rt[t, w, x] * m.eta_cha
        )
        if t == last_t:
            return end_of_hour_soc == m.E0
        return m.E_rt[next_t[t], w, x] == end_of_hour_soc
    m.soc_rt = pyo.Constraint(m.T, m.Omega, m.Xi, rule=soc_rt_rule)

    # SoC bounds
    def soc_min_rule(m, t, w, x):
        return m.E_rt[t, w, x] >= m.SoC_min * m.E_max
    m.soc_min_con = pyo.Constraint(m.T, m.Omega, m.Xi, rule=soc_min_rule)

    def soc_max_rule(m, t, w, x):
        return m.E_rt[t, w, x] <= m.SoC_max * m.E_max
    m.soc_max_con = pyo.Constraint(m.T, m.Omega, m.Xi, rule=soc_max_rule)

    # SoC cycling
    def soc_cycling_rule(m, t, w, x):
        if t in first_hour_each_day:
            return m.E_rt[t, w, x] == m.E0
        return pyo.Constraint.Skip
    m.soc_cycling = pyo.Constraint(m.T, m.Omega, m.Xi, rule=soc_cycling_rule)

    # # mFRR activation bounds
    # def mfrr_up_bound_rule(m, t, w, x):
    #     return m.a_up[t, w, x] <= (
    #         m.P_rt_grid[t, w, x] - (m.P_DA[t, w] + m.P_PPA_rt[t, w, x])
    #     )
    # m.mfrr_up_bound = pyo.Constraint(m.T, m.Omega, m.Xi, rule=mfrr_up_bound_rule)

    def mfrr_up_max_activation_limit_rule(m, t, w, x):
        activation_cap = min(
            float(pyo.value(m.P_grid)),
            float(pyo.value(m.A_up_max[t, w])),
        )
        return m.a_up[t, w, x]  <= m.y_up[t, w, x] * activation_cap
    m.mfrr_up_max_activation_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=mfrr_up_max_activation_limit_rule)

    def mfrr_down_max_activation_limit_rule(m, t, w, x):
        activation_cap = min(
            float(pyo.value(m.P_grid)),
            float(pyo.value(m.A_down_max[t, w])),
        )
        return m.a_down[t, w, x]  <= m.y_down[t, w, x] * activation_cap
    m.mfrr_down_max_activation_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=mfrr_down_max_activation_limit_rule)

    def mfrr_down_and_delta_bound_rule(m, t, w, x):
        return m.a_down[t, w, x] + m.delta_neg[t, w, x] <= m.P_DA[t, w]
    m.mfrr_down_bound = pyo.Constraint(m.T, m.Omega, m.Xi, rule=mfrr_down_and_delta_bound_rule)

    def mfrr_down_decompose_rule(m, t, w, x):
        return m.a_down[t, w, x] == m.P_cha_down[t, w, x] + m.curt_down[t, w, x]
    m.mfrr_down_decompose = pyo.Constraint(m.T, m.Omega, m.Xi, rule=mfrr_down_decompose_rule)

    def charging_decompose_rule(m, t, w, x):
        return m.P_cha_rt[t, w, x] == m.P_cha_free[t, w, x] + m.P_cha_down[t, w, x]
    m.charging_decompose = pyo.Constraint(m.T, m.Omega, m.Xi, rule=charging_decompose_rule)

    def curtailment_decompose_rule(m, t, w, x):
        return m.curt[t, w, x] == m.curt_free[t, w, x] + m.curt_down[t, w, x]
    m.curtailment_decompose = pyo.Constraint(m.T, m.Omega, m.Xi, rule=curtailment_decompose_rule)

    # Grid capacity
    def grid_rt_rule(m, t, w, x):
        return m.P_rt_grid[t, w, x] <= m.P_grid
    m.grid_rt = pyo.Constraint(m.T, m.Omega, m.Xi, rule=grid_rt_rule)

    # Curtailment cannot exceed available wind
    def curt_limit_rule(m, t, w, x):
        return m.curt[t, w, x] <= m.wind_real[t, w, x]
    m.curt_limit = pyo.Constraint(m.T, m.Omega, m.Xi, rule=curt_limit_rule)

    # Scenario profit definition
    def phi_rule(m, w, x):
        hourly_profit = sum(
            m.lambda_PPA * m.P_PPA[t]
            + m.lambda_spot[t, w] * m.P_DA[t, w]
            + m.lambda_imb[t, w, x] * m.delta[t, w, x]
            - m.c_imb * (m.delta_pos[t, w, x] + m.delta_neg[t, w, x])
            + m.a_up[t, w, x] * m.lambda_mfrr_up[t, w, x]
            - m.a_down[t, w, x] * m.lambda_mfrr_down[t, w, x]
            - m.degradation_cost * (m.P_dis_rt[t, w, x] + m.P_cha_rt[t, w, x])
            for t in m.T
        )
        settlement_shortfall_cost = m.c_ppa * sum(
            m.S_short[sp, w, x] for sp in m.SP
        )
        return m.Phi[w, x] == hourly_profit - settlement_shortfall_cost
    m.phi_def = pyo.Constraint(m.Omega, m.Xi, rule=phi_rule)

    # CVaR of profit.
    # Standard CVaR is usually written for losses.
    # Here we use profit, so we protect the lower tail by applying CVaR to (-Phi).
    def cvar_aux_rule(m, w, x):
        return m.u[w, x] >= -m.Phi[w, x] - m.eta_cvar
    m.cvar_aux = pyo.Constraint(m.Omega, m.Xi, rule=cvar_aux_rule)

    def cvar_expr(m):
        return -(
            m.eta_cvar
            + (1 / (1 - m.alpha)) * sum(
                m.pi_omega[w] * m.pi_xi_given_omega[w, x] * m.u[w, x]
                for w in m.Omega for x in m.Xi
            )
        )

    # Objective: (1-beta) E[Phi] + beta * CVaR_alpha
    def obj_rule(m):
        expected_profit = sum(
            m.pi_omega[w] * m.pi_xi_given_omega[w, x] * m.Phi[w, x]
            for w in m.Omega for x in m.Xi
        )
        return (1 - m.beta) * expected_profit + m.beta * cvar_expr(m)

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m
