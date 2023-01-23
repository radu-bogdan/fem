import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Define the grid
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)

# Define the function f(x, y)
f = lambda x, y: x**2 + y**2

# Define the boundary conditions
u_x0 = lambda y: 0
u_x1 = lambda y: y**2
u_y0 = lambda x: x**2
u_y1 = lambda x: 1

# Set the initial guess for the solution
u = np.zeros((101, 101))

# Set the maximum number of iterations
max_iter = 100

# Set the tolerance for the solution
tol = 1e-6

# Solve the Laplace problem
for i in range(max_iter):
  
  # Compute the right-hand side vector
  rhs = f(x[1:-1], y[1:-1])

  # Compute the boundary values
  u[0, 1:-1] = u_x0(y[1:-1])
  u[-1, 1:-1] = u_x1(y[1:-1])
  u[1:-1, 0] = u_y0(x[1:-1])
  u[1:-1, -1] = u_y1(x[1:-1])

  # Compute the coefficients of the linear system
  dx2, dy2 = 1 / (x[2] - x[1]), 1 / (y[2] - y[1])
  a = (dx2 + dy2) * np.ones(u[1:-1, 1:-1].shape)
  b = -dx2 * np.ones(u[1:-1, 1:-1].shape)
  c = -dy2 * np.ones(u[1:-1, 1:-1].shape)

  # Assemble the linear system
  A = csr_matrix((np.concatenate((a.flatten(), b.flatten(), c.flatten())),
                  (np.concatenate((np.arange(a.size), np.arange(a.size))),
                   np.concatenate((np.arange(a.size), np.arange(a.size) + 1, np.arange(a.size) + a.shape[1])))),
                 shape=(a.size, a.size))
  rhs = rhs.flatten()

  # Solve the linear system
  u[1:-1, 1:-1] = spsolve(A, rhs).reshape(a.shape)

  # Check the convergence of the solution
  if np.linalg.norm(u - u_prev) < tol:
    break

  # Store the current solution
  u_prev = u.copy()

# Print the number of iterations
print(f"Number of iterations: {i + 1}")

# Print the final solution
print(f"Final solution: \n{u}")