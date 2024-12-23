import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from Ej2 import quadratic_risk

print("---------- C1 ----------")

d_large = 200
n = 100
mu_large = np.zeros(d_large)
cov_large = np.eye(d_large)
X_large = np.random.multivariate_normal(mu_large, cov_large, size=n)
theta_star_large = np.array([10, 5, -3, -2, -1] + [0] * (d_large - 5))
epsilon = np.random.chisquare(df=1, size=n) - 1
Y_large = np.dot(X_large, theta_star_large) + epsilon

# Assuming X_large and Y_large are defined as in previous steps
n, d = X_large.shape  # X_large is n x d, n=100, d=200
empirical_risk = []

# Loop through models M_d with 1 <= d <= 200
for i in range(1, d + 1):
    X_d = X_large[:, :i]  # Get the first i columns of X_large
    model = LinearRegression().fit(X_d, Y_large)
    theta_d = model.coef_
    residuals = Y_large - model.predict(X_d)
    risk_d = np.mean(residuals**2)  # Empirical risk is the MSE
    empirical_risk.append(risk_d)

# Plot the empirical risk vs. d
plt.plot(range(1, d + 1), empirical_risk)
plt.xlabel('d (Number of features)')
plt.ylabel('Empirical Risk')
plt.title('Empirical Risk vs. Number of Features (d)')
plt.show()

# As d (the number of features) increases, the empirical risk tends to decrease. 
# This happens because more features allow the model to better fit the data, 
# which reduces the residuals (the difference between the predicted and actual values).

print("---------- C2 ----------")

# Calcular el riesgo cuadrático para cada modelo M_d
risks = []
for d in range(1, d_large + 1):  # Solo hasta 50 columnas de X
    X_d = X_large[:, :d]  # Usar las primeras d columnas de X
    risk, _ = quadratic_risk(X_d, Y_large, theta_star_large[:d])
    risks.append(risk)



print("---------- C3 ----------")

# Mallow's C_p criterion
sigma_squared = np.var(Y_large - model.predict(X_large))  # Estimate residual variance

# Calcular el riesgo cuadrático para cada modelo M_d
cp_values = []
for d in range(1, d_large + 1):  # Solo hasta 50 columnas de X
    X_d = X_large[:, :d]  # Usar las primeras d columnas de X
    penalty = 2 * i * sigma_squared / n
    cp_value, _ = quadratic_risk(X_d, Y_large, theta_star_large[:d], penalty=penalty)
    cp_values.append(cp_value)

# Plot quadratic risk vs. d
plt.plot(range(1, d + 1), risks, label="Quadratic Risk")
plt.plot(range(1, d + 1), cp_values, label="Mallow's C_p")
plt.xlabel('d (Number of features)')
plt.ylabel('Quadratic Risk')
plt.title('Quadratic Risk vs. Number of Features (d)')
plt.legend()
plt.show()

print("---------- C4 ----------")

# Function for Hold-out cross-validation
def ho_cross_validation(X, y, p, model):
    n = len(X)
    train_size = n - p
    errors = []
    
    for _ in range(100):  # Repeat 100 times for better estimate
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        model.fit(X_train, y_train)
        model.predict(X_test)
        y_pred = np.dot(X_test, model.coef_)
        error = np.mean((y_test - y_pred) ** 2)
        errors.append(error)
    
    return np.mean(errors)

# Calculate HO(p) for different p values
p_values = [20, 50, 80]
ho_errors = {p: [] for p in p_values}

for p in p_values:
    for i in range(1, d + 1):
        X_d = X_large[:, :i]
        model = LinearRegression()
        error = ho_cross_validation(X_d, Y_large, p, model)
        ho_errors[p].append(error)

# Plot HO(p) for different p values
for p in p_values:
    plt.plot(range(1, d + 1), ho_errors[p], label=f'HO(p={p})')

plt.xlabel('d (Number of features)')
plt.ylabel('Hold-out Cross-validation Error')
plt.title('Hold-out Cross-validation Error vs. Number of Features (d)')
plt.legend()
plt.ylim(0, 200)
plt.show()

print("---------- C4.2 ----------")


# Function for V-fold cross-validation
def v_fold_cv(X, y, V, model):
    scores = cross_val_score(model, X, y, cv=V, scoring='neg_mean_squared_error')
    return -np.mean(scores)  # Convert to positive MSE

# Calculate V-FCV for different V values
V_values = [2, 5, 10]
v_errors = {V: [] for V in V_values}

for V in V_values:
    for i in range(1, d + 1):
        X_d = X_large[:, :i]
        model = LinearRegression()
        error = v_fold_cv(X_d, Y_large, V, model)
        v_errors[V].append(error)

# Plot V-FCV for different V values
for V in V_values:
    plt.plot(range(1, d + 1), v_errors[V], label=f'V-FCV (V={V})')

plt.xlabel('d (Number of features)')
plt.ylabel('V-Fold Cross-validation Error')
plt.title('V-Fold Cross-validation Error vs. Number of Features (d)')
plt.legend()
plt.ylim(0, 1000)
plt.show()