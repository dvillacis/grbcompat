"""
Sudoku Solver via Integer Programming
======================================
Solve any 9x9 Sudoku puzzle by encoding it as a binary MIP:

  x[i, j, v] = 1  iff cell (i,j) contains digit v   (i,j,v ∈ 1..9)

Constraints make this equivalent to standard Sudoku rules:
  • Each cell holds exactly one digit.
  • Each digit appears exactly once in every row.
  • Each digit appears exactly once in every column.
  • Each digit appears exactly once in every 3x3 box.
  • Fixed (given) cells are pinned to their known value.

The objective is constant (all solutions are equally "optimal"), so
HiGHS finds a feasible integer point — a perfect constraint-satisfaction demo.

Concepts: 3-D addVars, addConstrs with generators, constraint as equality.
"""

import grbcompat as gp
from grbcompat import GRB

# 0 = empty cell
PUZZLE = [
    [5, 3, 0,  0, 7, 0,  0, 0, 0],
    [6, 0, 0,  1, 9, 5,  0, 0, 0],
    [0, 9, 8,  0, 0, 0,  0, 6, 0],

    [8, 0, 0,  0, 6, 0,  0, 0, 3],
    [4, 0, 0,  8, 0, 3,  0, 0, 1],
    [7, 0, 0,  0, 2, 0,  0, 0, 6],

    [0, 6, 0,  0, 0, 0,  2, 8, 0],
    [0, 0, 0,  4, 1, 9,  0, 0, 5],
    [0, 0, 0,  0, 8, 0,  0, 7, 9],
]

DIGITS = range(1, 10)
CELLS  = range(9)


def _box(i, j):
    """Return the 3×3 box index (0-8) for cell (i, j)."""
    return (i // 3) * 3 + (j // 3)


def print_grid(grid, title=""):
    if title:
        print(f"\n  {title}")
    print("  ╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗")
    for i in CELLS:
        row = "  ║"
        for j in CELLS:
            val = grid[i][j]
            row += (f" {val} " if val else " · ") + ("║" if j in (2, 5, 8) else "│")
        print(row)
        if i in (2, 5):
            print("  ╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣")
        elif i < 8:
            print("  ╟───┼───┼───╫───┼───┼───╫───┼───┼───╢")
    print("  ╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝")


def main():
    m = gp.Model("sudoku")
    m.Params.OutputFlag = 0

    # x[i, j, v] = 1 iff cell (i,j) holds digit v
    x = m.addVars(
        [(i, j, v) for i in CELLS for j in CELLS for v in DIGITS],
        vtype=GRB.BINARY,
        name="x",
    )

    # Dummy objective — any feasible solution is acceptable
    m.setObjective(0, GRB.MINIMIZE)

    # Each cell holds exactly one digit
    m.addConstrs(
        (x.sum(i, j, "*") == 1 for i in CELLS for j in CELLS),
        name="one_digit",
    )

    # Each digit appears exactly once per row
    m.addConstrs(
        (x.sum(i, "*", v) == 1 for i in CELLS for v in DIGITS),
        name="row",
    )

    # Each digit appears exactly once per column
    m.addConstrs(
        (x.sum("*", j, v) == 1 for j in CELLS for v in DIGITS),
        name="col",
    )

    # Each digit appears exactly once per 3×3 box
    for b in range(9):
        cells_in_box = [(i, j) for i in CELLS for j in CELLS if _box(i, j) == b]
        for v in DIGITS:
            m.addConstr(
                gp.quicksum(x[i, j, v] for i, j in cells_in_box) == 1,
                name=f"box_{b}_d{v}",
            )

    # Pin given cells
    for i in CELLS:
        for j in CELLS:
            if PUZZLE[i][j] != 0:
                m.addConstr(x[i, j, PUZZLE[i][j]] == 1, name=f"given_{i}_{j}")

    m.optimize()

    print("=" * 47)
    print("  Sudoku Solver — Integer Programming")
    print("=" * 47)
    print_grid(PUZZLE, "Puzzle (· = empty):")

    if m.Status != GRB.OPTIMAL:
        print(f"\n  No solution found (status {m.Status})")
        return

    solution = [
        [next(v for v in DIGITS if x[i, j, v].X > 0.5) for j in CELLS]
        for i in CELLS
    ]
    print_grid(solution, "Solution:")

    # Verify
    errors = []
    for i in CELLS:
        for j in CELLS:
            if PUZZLE[i][j] and PUZZLE[i][j] != solution[i][j]:
                errors.append((i, j))
    status = "✓ All given cells preserved." if not errors else f"✗ {len(errors)} errors."
    print(f"\n  {status}")
    print(f"  Solved in {m.Runtime:.3f}s using {m.NumVars} binary variables.")


if __name__ == "__main__":
    main()
