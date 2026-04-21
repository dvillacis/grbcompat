"""
Nurse Shift Scheduling
======================
Schedule 7 nurses across 7 days × 2 shifts (Day / Night) so that:
  • Minimum staffing levels are met every shift.
  • No nurse works both the day and night shift on the same day.
  • Each nurse works at most 5 shifts per week.
  • Each nurse gets at least 2 days completely off.
  • Night shifts are balanced: no nurse works more than 3 nights.

The objective minimises the total number of shift assignments (i.e.
achieves the schedule with the fewest nurses working, which also
minimises overtime risk).

Concepts: 3-D binary addVars, addConstrs with generators, tupledict.sum().
"""

import grbcompat as gp
from grbcompat import GRB

NURSES = [f"Nurse_{i+1}" for i in range(7)]
DAYS   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["Day", "Night"]

# Minimum nurses required per (day, shift)
MIN_STAFF = {
    ("Mon", "Day"): 3, ("Mon", "Night"): 2,
    ("Tue", "Day"): 3, ("Tue", "Night"): 2,
    ("Wed", "Day"): 3, ("Wed", "Night"): 2,
    ("Thu", "Day"): 3, ("Thu", "Night"): 2,
    ("Fri", "Day"): 3, ("Fri", "Night"): 2,
    ("Sat", "Day"): 2, ("Sat", "Night"): 2,
    ("Sun", "Day"): 2, ("Sun", "Night"): 2,
}

MAX_SHIFTS_PER_WEEK = 5
MAX_NIGHTS_PER_WEEK = 3
MIN_DAYS_OFF        = 2


def main():
    m = gp.Model("nurse_scheduling")
    m.Params.OutputFlag = 0

    # x[nurse, day, shift] = 1 if that nurse works that shift
    x = m.addVars(
        [(n, d, s) for n in NURSES for d in DAYS for s in SHIFTS],
        vtype=GRB.BINARY,
        name="work",
    )

    # Minimise total assignments
    m.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)

    # Minimum staffing per shift
    for d in DAYS:
        for s in SHIFTS:
            m.addConstr(
                x.sum("*", d, s) >= MIN_STAFF[d, s],
                name=f"staff_{d}_{s}",
            )

    # No nurse works both shifts on the same day
    for n in NURSES:
        for d in DAYS:
            m.addConstr(
                x.sum(n, d, "*") <= 1,
                name=f"one_shift_{n}_{d}",
            )

    # Weekly shift cap
    for n in NURSES:
        m.addConstr(
            x.sum(n, "*", "*") <= MAX_SHIFTS_PER_WEEK,
            name=f"max_shifts_{n}",
        )

    # Night shift cap
    for n in NURSES:
        m.addConstr(
            x.sum(n, "*", "Night") <= MAX_NIGHTS_PER_WEEK,
            name=f"max_nights_{n}",
        )

    # Minimum days off (a day off = no shift that day)
    for n in NURSES:
        days_working = gp.quicksum(x[n, d, s] for d in DAYS for s in SHIFTS)
        m.addConstr(
            days_working <= len(DAYS) - MIN_DAYS_OFF,
            name=f"days_off_{n}",
        )

    m.optimize()

    print("=" * 56)
    print("  Nurse Shift Schedule")
    print("=" * 56)

    if m.Status != GRB.OPTIMAL:
        print(f"Solver status: {m.Status}")
        return

    print(f"\n  Total assignments: {int(m.ObjVal)}")
    print(f"  Nurses: {len(NURSES)}   Days: {len(DAYS)}   Shifts: {len(SHIFTS)}\n")

    # Print schedule grid
    col_w = 10
    header = f"  {'Nurse':<12}" + "".join(f"{d:>{col_w}}" for d in DAYS)
    print(header)
    print("  " + "─" * (12 + len(DAYS) * col_w))

    for n in NURSES:
        row = f"  {n:<12}"
        nights = 0
        total  = 0
        for d in DAYS:
            day_on   = x[n, d, "Day"].X   > 0.5
            night_on = x[n, d, "Night"].X > 0.5
            if day_on:
                cell = "Day"
                total += 1
            elif night_on:
                cell = "Night"
                total += 1
                nights += 1
            else:
                cell = "OFF"
            row += f"{cell:>{col_w}}"
        row += f"   ({total} shifts, {nights} nights)"
        print(row)

    print("\n  Staffing coverage check:")
    print(f"  {'Shift':<18} {'Required':>9} {'Assigned':>9}")
    print(f"  {'─'*18}  {'─'*9}  {'─'*9}")
    for d in DAYS:
        for s in SHIFTS:
            assigned = sum(
                1 for n in NURSES if x[n, d, s].X > 0.5
            )
            print(f"  {d+' '+s:<18} {MIN_STAFF[d,s]:>9} {assigned:>9}")


if __name__ == "__main__":
    main()
