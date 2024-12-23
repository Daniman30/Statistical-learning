import numpy as np
import matplotlib.pyplot as plt

# A.1: Draw n=100 Gaussian vectors Xi from N_d(μ, I_d), with μ=0 and d=50
n = 100
d = 50
mu = np.zeros(d)
cov = np.eye(d)  # Identity matrix for I_d
X = np.random.multivariate_normal(mu, cov, size=n)

# A.2: Create a vector ε with n=100 coordinates, centered, with Chi-square χ²(1)
epsilon = np.random.chisquare(df=1, size=n) - 1  # Centering by subtracting the mean (1)

# A.3: Define a parameter vector θ* = (10, 5, -3, -2, -1, 0,…, 0)^T with dimension d=50
theta_star = np.array([10, 5, -3, -2, -1] + [0] * (d - 5))

# A dependency of Y on the influential variables for each zeta value* different from zero is created
# Influential variables are the non-zero coefficients in θ* (i.e., the first five elements)

# A.4: Create a vector Y = X θ* + ε of dimension n=100
Y = np.dot(X, theta_star) + epsilon

# A.5: Plot the graph of Y versus the third coordinate X^3
plt.figure(figsize=(12, 6))

# Y vs X^3
plt.subplot(1, 2, 1)
plt.scatter(X[:, 2], Y, color='blue', alpha=0.7)
plt.title("Y vs X^3")
plt.xlabel("X^3")
plt.ylabel("Y")

# Y vs X^10
plt.subplot(1, 2, 2)
plt.scatter(X[:, 9], Y, color='green', alpha=0.7)
plt.title("Y vs X^10")
plt.xlabel("X^10")
plt.ylabel("Y")

plt.tight_layout()
plt.show()

# Observations:
# - The graph of Y vs X^3 shows a little relationship because θ_3 = -3 (non-zero).
# - The graph of Y vs X^10 shows no clear relationship because θ_10 = 0 (zero).
