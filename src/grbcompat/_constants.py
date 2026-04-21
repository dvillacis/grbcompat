class GRB:
    """Constants mirroring the gurobipy GRB namespace."""

    INFINITY = 1e100
    UNDEFINED = 1e101  # sentinel for "no MIP start hint set"

    # Objective sense
    MINIMIZE = 1
    MAXIMIZE = -1

    # Variable types
    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"
    SEMICONT = "S"
    SEMIINT = "N"

    # Constraint senses
    LESS_EQUAL = "<"
    GREATER_EQUAL = ">"
    EQUAL = "="

    # SOS types
    SOS_TYPE1: int = 1
    SOS_TYPE2: int = 2

    # Solve-status codes (top-level, mirroring gurobipy)
    LOADED = 1
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    CUTOFF = 6
    ITERATION_LIMIT = 7
    NODE_LIMIT = 8
    TIME_LIMIT = 9
    SOLUTION_LIMIT = 10
    INTERRUPTED = 11
    NUMERIC = 12
    SUBOPTIMAL = 13
    INPROGRESS = 14
    USER_OBJ_LIMIT = 15

    class Status:
        LOADED = 1
        OPTIMAL = 2
        INFEASIBLE = 3
        INF_OR_UNBD = 4
        UNBOUNDED = 5
        CUTOFF = 6
        ITERATION_LIMIT = 7
        NODE_LIMIT = 8
        TIME_LIMIT = 9
        SOLUTION_LIMIT = 10
        INTERRUPTED = 11
        NUMERIC = 12
        SUBOPTIMAL = 13
        INPROGRESS = 14
        USER_OBJ_LIMIT = 15

    class Attr:
        ObjVal = "ObjVal"
        X = "X"
        Pi = "Pi"
        Slack = "Slack"
        RC = "RC"
        Status = "Status"
        VarName = "VarName"
        ConstrName = "ConstrName"
        LB = "LB"
        UB = "UB"
        Obj = "Obj"
        VType = "VType"
        RHS = "RHS"
        Sense = "Sense"

    class Callback:
        """gurobipy-compatible callback where-codes and what-codes."""

        # ---- where codes ----
        POLLING  = 0
        PRESOLVE = 1
        SIMPLEX  = 6
        MIP      = 7
        MIPSOL   = 8    # new MIP feasible solution found
        MIPNODE  = 9    # LP relaxation solved at a B&B node
        MESSAGE  = 10
        BARRIER  = 11

        # ---- what codes for MIPSOL context ----
        MIPSOL_OBJ    = 10001   # objective value of the current solution
        MIPSOL_OBJBST = 10002   # best primal bound found so far
        MIPSOL_OBJBND = 10003   # current best dual bound
        MIPSOL_NODCNT = 10004   # number of explored B&B nodes (int)
        MIPSOL_SOLCNT = 10005   # number of feasible solutions found so far
