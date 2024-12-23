Make these exercises with Python (Spyder is the preferred software if possible).
A. Generating random variables:
1. Draw $n=100$ Gaussian vectors $X_i$ from $N_d( \mu,I_d)$, with $\mu=0$ and $d=50$.

2. Create a vector $\epsilon$ with $n=100$ coordinates, centered, with a Chi-square $\chi^2(1)$ with one degree of freedom probability distribution.

3. Define a parameter vector $\theta^\star = (10, 5, -3, -2, -1, 0,…, 0)^T$ with dimension $d=50$.
   1. What can you conclude regarding the influential variables?
4. Create a vector $Y = X  \theta^\star + \epsilon$ of dimension $n=100$.
   1. The purpose is now to estimate the unknown vector $\theta^\star$.
5. Plot the graph of $Y$ versus the third coordinate $X^3$.
    1. Compare with the graph of $Y$ versus $X^10$. What can you see?

B. Linear regression model:
1. Import the Scikit Learn Python library and extract the LinearRegression function from it.
2. Fit the linear regression model on the data with an intercept and output the values of all estimated coefficients. 
   1. Can you conclude anything? 
   2. What is the rank of the design matrix $X$?
3. Write a function computing the quadratic risk of the least-squares estimator $\hat \theta$.
4. Reproduce the previous steps from A.1 with $\mu=0$ for the first 5 features and $\mu=10$ for all the other ones. 
   1. What can see about the estimated coefficients?
5. Reproduce the steps from A.1 now with $d=200$. 
   1. After estimating the coefficients, what do you see?
   2. What is the rank of the design matrix $X$?

C. Model selection:
1. Consider a sequence of nested linear models denoted by $(M_d)_{1\leq d\leq D}$ with $D=200$, where each model $M_d$ is made of the first $d$ columns of the full design matrix $X$.
   1. Write a code calculating the empirical risk $\hat R(\hat \theta_d)$ of the least squares estimator for each model $M_d$ and draw the graph of the function $d \mapsto \hat R(\hat \theta_d)$.
   2. What can you see?
2. Display the graph of the function $d \mapsto Risk( \hat \theta_d)$, where $Risk( \hat \theta_d)$ denotes the quadratic risk of the least-squares estimator $\hat \theta_d$.
3. On the same figure, draw the graph of Mallow’s C_p criterion and make the comparison with the empirical risk and the risk curves.
4. Implement a function computing the hold-$p$-out HO(p) cross-validation (CV) criterion with $p$ denoting the cardinality of the test set. 
   1. Compare the curve of HO(p) for the different models with $p=20$, $p=50$, and $p=80$.
      1. What do you see?
   2. Repeat the same process with the $V$-fold CV (V-FCV) with $V=2$, $V=5$, and $V=10$.
      1. Could you draw any conclusion?



