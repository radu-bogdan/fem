from scipy.sparse.linalg import onenormest, splu, LinearOperatordef condest(A):    luA = splu(A)    iA = LinearOperator(luA.shape, matvec = lambda x : luA.solve(x), rmatvec = lambda x : luA.solve(x))    return onenormest(iA)*onenormest(A)