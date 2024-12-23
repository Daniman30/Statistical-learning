from sklearn.linear_model import LinearRegression
import numpy as np
import Ej1

print("---------- B1 ----------")

# B.1 Import LinearRegression and fit the model
model = LinearRegression()
model.fit(Ej1.X, Ej1.Y)

print("---------- B2 ----------")

# B.2 Output the estimated coefficients
theta_hat = model.coef_
intercept_hat = model.intercept_

print("Estimated coefficients (θ̂):", theta_hat)
print("Estimated intercept:", intercept_hat)

# B.2.1 Observations
# The first 5 variables in the coefficient vector have larger absolute values, 
# suggesting that these variables can be the most influential 
# coinciding with the influential variables.

# B.2.2 Rank of the design matrix
rank_X = np.linalg.matrix_rank(Ej1.X)
print("Size of X: ", len(Ej1.X), "x", len(Ej1.X[0]))
print("Rank of the design matrix X:", rank_X)

# The matrix is full rank, this implies that all characteristics in X 
# are linearly independent of each other

print("---------- B3 ----------")

def quadratic_risk(X, Y, theta_star, num_simulations=1000, penalty=0):
    """
    Computes the quadratic risk (Rn) of the least squares estimator θ^.

    Parameters:
    - X: design matrix of dimension (n, d)
    - Y: output vector (n,)
    - theta_star: vector of true parameters θ* of dimension (d,)
    - num_simulations: number of simulations to approximate the expectation

    Returns:
    - quadratic_risk: the estimated quadratic risk
    """
    n, d = X.shape

    # Fit the model to obtain θ^ (least squares estimator)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    theta_hat = model.coef_  # Estimated coefficients (θ^)

    # List to store quadratic losses in each simulation
    quadratic_losses = []

    for _ in range(num_simulations):
        # Generate new errors ε' ~ χ²(1) - 1
        epsilon_new = np.random.chisquare(df=1, size=n) - 1

        # Create new Y' values ​​using the same X and θ*
        Y_new = np.dot(X, theta_star) + epsilon_new

        # Calculate the quadratic loss for this new data set
        loss = np.mean((Y_new - np.dot(X, theta_hat)) ** 2) + penalty
        quadratic_losses.append(loss)

    # Average the quadratic losses to approximate the quadratic risk
    return np.mean(quadratic_losses), quadratic_losses

print("Quadratic Risk: ", quadratic_risk(Ej1.X, Ej1.Y, Ej1.theta_star)[0])

print("---------- B4 ----------")

# B.4: Modify μ for the first 5 features to 0, others to 10
mu_modified = np.array([0]*5 + [10]*(Ej1.d-5))
X_modified = np.random.multivariate_normal(mu_modified, Ej1.cov, size=Ej1.n)
Y_modified = X_modified @ Ej1.theta_star + Ej1.epsilon

model_modified = LinearRegression(fit_intercept=True)
model_modified.fit(X_modified, Y_modified)
estimated_coefficients_modified = model_modified.coef_

print("Modified Estimated Coefficients:", estimated_coefficients_modified)

# The first few coefficients (for $\theta_1, \theta_2, \theta_3, \theta_4, \theta_5$) 
# are relatively close to the true values of $\theta^\star = [10, 5, -3, -2, -1]$.
#  For example:
# $\theta_1 \approx 9.99$ (close to 10)
# $\theta_2 \approx 4.74$ (close to 5)
# $\theta_3 \approx -2.85$ (close to -3)
# $\theta_4 \approx -1.97$ (close to -2)
# $\theta_5 \approx -1.40$ (close to -1)
# These results indicate that the model is correctly identifying the major influencing variables and their approximate magnitudes.

# The coefficients for the features where the mean was set to 10 (features 6 through 50) 
# have values that are much smaller in magnitude, often close to zero (e.g., 0.072, -0.059, -0.073, etc.).
# This suggests that, due to the modification of the means of these features, 
# the estimated coefficients for these features are less significant. 
# Since the corresponding values of $\theta^\star$ for these variables were set to 0, 
# the model correctly identifies their minimal influence on the output $Y$.

print("---------- B5 ----------")

# B.5: Increase dimensionality to d=200
d_large = 200
mu_large = np.zeros(d_large)
cov_large = np.eye(d_large)
X_large = np.random.multivariate_normal(mu_large, cov_large, size=Ej1.n)
theta_star_large = np.array([10, 5, -3, -2, -1] + [0] * (d_large - 5))
Y_large = X_large @ theta_star_large + Ej1.epsilon

model_large = LinearRegression(fit_intercept=True)
model_large.fit(X_large, Y_large)
estimated_coefficients_large = model_large.coef_

print("Large Dimension Estimated Coefficients:", estimated_coefficients_large)
print("Rank of X_large:", np.linalg.matrix_rank(X_large))

# The estimated coefficients show a more varied spread, with values ranging from large positive 
# to large negative numbers. This reflects the increased complexity of the design matrix when 
# the number of features grows.

# The large number of features (200) introduces more noise, which can result in slight overfitting 
# or instability in the coefficients. As a result, some coefficients might fluctuate around zero 
# even for features with no true effect. This can lead to higher variance in the estimated coefficients.

