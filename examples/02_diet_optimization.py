"""
Diet Optimization — Minimum-Cost Nutrition
==========================================
Find the cheapest daily menu that satisfies all nutritional targets.
After solving, shadow prices (dual variables) reveal the *marginal cost*
of tightening each nutritional requirement by one unit — a great way to
understand LP duals in a concrete setting.

Concepts: LP, dual values (Pi), shadow price interpretation, getAttr batch read.
"""

import grbcompat as gp
from grbcompat import GRB

# Food: (cost $/serving, calories, protein g, fat g, carbs g, fiber g)
FOODS = {
    "oatmeal":       (0.50, 303, 13, 5.5, 51,  8),
    "whole_milk":    (0.60, 149,  8, 8.0, 12,  0),
    "eggs":          (0.40, 154, 13, 11,   1,  0),
    "bread":         (0.30, 120,  4, 1.5, 23,  1),
    "chicken_breast":(1.20, 165, 31, 3.5,   0,  0),
    "brown_rice":    (0.35, 216,  5, 1.8, 45,  3),
    "broccoli":      (0.45,  55,  4, 0.6,  11,  5),
    "almonds":       (0.80, 164,  6, 14,   6,  4),
    "banana":        (0.25,  89,  1, 0.3, 23,  3),
    "canned_tuna":   (0.90, 109, 25, 2.5,   0,  0),
    "lentils":       (0.40, 230, 18, 0.8, 40, 16),
    "olive_oil":     (0.60, 119,  0, 14,   0,  0),
}

# Daily nutritional requirements (lower bounds)
REQUIREMENTS = {
    "calories": 2000,
    "protein":    50,  # g
    "fat":        44,  # g
    "carbs":     260,  # g
    "fiber":      25,  # g
}

# Column indices into FOODS tuple
IDX = {"calories": 1, "protein": 2, "fat": 3, "carbs": 4, "fiber": 5}


def main():
    foods = list(FOODS)
    n = len(foods)

    m = gp.Model("diet")
    m.Params.OutputFlag = 0

    # Servings of each food (continuous, >= 0)
    x = m.addVars(n, lb=0.0, name=[f for f in foods])

    # Minimize daily cost
    cost = [FOODS[f][0] for f in foods]
    m.setObjective(gp.quicksum(cost[i] * x[i] for i in range(n)), GRB.MINIMIZE)

    # Nutritional constraints
    constrs = {}
    for nutrient, req in REQUIREMENTS.items():
        col = IDX[nutrient]
        content = [FOODS[f][col] for f in foods]
        constrs[nutrient] = m.addConstr(
            gp.quicksum(content[i] * x[i] for i in range(n)) >= req,
            name=nutrient,
        )

    # At most 3 servings of any single food (variety)
    for i in range(n):
        m.addConstr(x[i] <= 3.0, name=f"max_{foods[i]}")

    m.optimize()

    print("=" * 50)
    print("  Diet Optimization — Minimum-Cost Daily Menu")
    print("=" * 50)

    if m.Status != GRB.OPTIMAL:
        print(f"Solver status: {m.Status}")
        return

    print(f"\n  Total daily cost: ${m.ObjVal:.2f}\n")

    print(f"  {'Food':<18} {'Servings':>9}")
    print(f"  {'─'*18}  {'─'*9}")
    for i, food in enumerate(foods):
        if x[i].X > 0.01:
            print(f"  {food:<18} {x[i].X:>9.2f}")

    print(f"\n  Shadow prices — cost ($/unit) of tightening each requirement:")
    print(f"  {'Nutrient':<12} {'Requirement':>12} {'Shadow price':>13}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*13}")
    for nutrient, constr in constrs.items():
        unit = "kcal" if nutrient == "calories" else "g"
        print(
            f"  {nutrient:<12} {REQUIREMENTS[nutrient]:>10}{unit:>3}  "
            f"${-constr.Pi:>11.6f}"
        )

    print(
        "\n  Interpretation: the shadow price is the extra daily cost if you"
        "\n  raise that requirement by 1 unit (e.g. +1 g protein costs "
        f"${-constrs['protein'].Pi:.4f})."
    )


if __name__ == "__main__":
    main()
