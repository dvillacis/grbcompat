"""
Coffee Supply Chain — Multi-Echelon Network Flow
=================================================
Three coffee farms ship green beans to two roasteries, which then ship
roasted coffee to four cafés.  Minimise total shipping + roasting cost
while meeting café demand and respecting farm supply and roastery
throughput limits.

Network:
    Farms  ──►  Roasteries  ──►  Cafés
    (supply)    (capacity,        (demand)
                roasting cost)

Concepts: multi-index addVars, tupledict.sum(), 2-stage network LP.
"""

import grbcompat as gp
from grbcompat import GRB

# ── Data ─────────────────────────────────────────────────────────────

FARMS      = ["Yirgacheffe", "Huila",  "Antigua"]
ROASTERIES = ["Portland",    "Austin"]
CAFES      = ["Downtown", "Airport", "University", "Suburb"]

# Farm supply (bags/month)
SUPPLY = {"Yirgacheffe": 200, "Huila": 300, "Antigua": 250}

# Café demand (bags/month)
DEMAND = {"Downtown": 120, "Airport": 80, "University": 150, "Suburb": 100}

# Roastery throughput capacity (bags/month)
CAPACITY = {"Portland": 300, "Austin": 350}

# Shipping cost: farm → roastery  ($/bag)
SHIP_FR = {
    ("Yirgacheffe", "Portland"): 3.2,
    ("Yirgacheffe", "Austin"):   5.1,
    ("Huila",       "Portland"): 4.8,
    ("Huila",       "Austin"):   2.9,
    ("Antigua",     "Portland"): 6.0,
    ("Antigua",     "Austin"):   3.5,
}

# Roasting + shipping cost: roastery → café  ($/bag)
SHIP_RC = {
    ("Portland",  "Downtown"):    1.5,
    ("Portland",  "Airport"):     2.0,
    ("Portland",  "University"):  2.5,
    ("Portland",  "Suburb"):      3.0,
    ("Austin",    "Downtown"):    3.5,
    ("Austin",    "Airport"):     2.8,
    ("Austin",    "University"):  1.8,
    ("Austin",    "Suburb"):      2.2,
}


def main():
    m = gp.Model("coffee_supply_chain")
    m.Params.OutputFlag = 0

    # Variables: flow on each arc
    x_fr = m.addVars(SHIP_FR.keys(), lb=0.0, name="farm_to_roastery")
    x_rc = m.addVars(SHIP_RC.keys(), lb=0.0, name="roastery_to_cafe")

    # Objective: minimise total cost
    m.setObjective(
        gp.quicksum(SHIP_FR[k] * x_fr[k] for k in SHIP_FR)
        + gp.quicksum(SHIP_RC[k] * x_rc[k] for k in SHIP_RC),
        GRB.MINIMIZE,
    )

    # Farm supply limits
    for f in FARMS:
        m.addConstr(
            x_fr.sum(f, "*") <= SUPPLY[f],
            name=f"supply_{f}",
        )

    # Roastery capacity limits
    for r in ROASTERIES:
        m.addConstr(
            x_fr.sum("*", r) <= CAPACITY[r],
            name=f"capacity_{r}",
        )

    # Flow balance at roasteries: in == out
    for r in ROASTERIES:
        m.addConstr(
            x_fr.sum("*", r) == x_rc.sum(r, "*"),
            name=f"balance_{r}",
        )

    # Café demand satisfaction
    for c in CAFES:
        m.addConstr(
            x_rc.sum("*", c) >= DEMAND[c],
            name=f"demand_{c}",
        )

    m.optimize()

    print("=" * 55)
    print("  Coffee Supply Chain — Minimum-Cost Network Flow")
    print("=" * 55)

    if m.Status != GRB.OPTIMAL:
        print(f"Solver status: {m.Status}")
        return

    print(f"\n  Optimal monthly shipping cost: ${m.ObjVal:,.2f}\n")

    print("  Farm → Roastery flows (bags/month):")
    print(f"  {'Route':<32} {'Flow':>6}  {'Cost':>8}")
    print(f"  {'─'*32}  {'─'*6}  {'─'*8}")
    for (f, r), cost in SHIP_FR.items():
        flow = x_fr[f, r].X
        if flow > 0.5:
            print(f"  {f} → {r:<18} {flow:>6.0f}  ${flow*cost:>7.2f}")

    print("\n  Roastery → Café flows (bags/month):")
    print(f"  {'Route':<30} {'Flow':>6}  {'Cost':>8}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*8}")
    for (r, c), cost in SHIP_RC.items():
        flow = x_rc[r, c].X
        if flow > 0.5:
            print(f"  {r} → {c:<20} {flow:>6.0f}  ${flow*cost:>7.2f}")

    print("\n  Roastery utilisation:")
    for r in ROASTERIES:
        used = x_fr.sum("*", r).getValue()
        pct  = 100 * used / CAPACITY[r]
        print(f"    {r:<10}: {used:.0f}/{CAPACITY[r]} bags  ({pct:.0f}% utilised)")


if __name__ == "__main__":
    main()
