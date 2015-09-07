"""Bayesian Data Analysis, 3rd ed
PyStan demo

Demo for using Stan with Python interface PyStan.

"""

import numpy as np
import pystan
import matplotlib.pyplot as plt

# edit default plot settings (colours from colorbrewer2.org)
plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))

# ====== Bernoulli model =======================================================
bernoulli_code = """
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  for (n in 1:N)
    y[n] ~ bernoulli(theta);
}
"""
data = dict(N=10, y=[0,1,0,0,1,1,1,0,1,0])
fit = pystan.stan(model_code=bernoulli_code, data=data)
print(fit)
samples = fit.extract(permuted=True)
plt.hist(samples['theta'], 50)
plt.show()


# ====== Vectorized Bernoulli model ============================================
bernoulli_code = """
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  y ~ bernoulli(theta);
}
"""
data = dict(N=10, y=[1,1,1,0,1,1,1,0,1,1])
fit = pystan.stan(model_code=bernoulli_code, data=data)


# ====== Binomial model ========================================================
binomial_code = """
data {
  int<lower=0> N;
  int<lower=0> y;
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  y ~ binomial(N,theta);
}
"""
data = dict(N=10, y=8)
fit = pystan.stan(model_code=binomial_code, data=data)
samples = fit.extract(permuted=True)
plt.hist(samples['theta'], 50)
plt.show()

# ====== Re-running Binomial model with new data ===============================
data = dict(N=10, y=10)
fit = pystan.stan(fit=fit, data=data)
samples = fit.extract(permuted=True)
plt.hist(samples['theta'], 50)
plt.show()


# ====== Comparison of two groups with Binomial ================================
binomial_code = """
data {
  int<lower=0> N1;
  int<lower=0> y1;
  int<lower=0> N2;
  int<lower=0> y2;
}
parameters {
  real<lower=0,upper=1> theta1;
  real<lower=0,upper=1> theta2;
}
transformed parameters {
  real oddsratio;
  oddsratio <- (theta2/(1-theta2))/(theta1/(1-theta1));
}
model {
  theta1 ~ beta(1,1);
  theta2 ~ beta(1,1);
  y1 ~ binomial(N1,theta1);
  y2 ~ binomial(N2,theta2);
}
"""
data = dict(N1=674, y1=39, N2=680, y2=22)
fit = pystan.stan(model_code=binomial_code, data=data)
samples = fit.extract(permuted=True)
plt.hist(samples['oddsratio'], 50)
plt.show()


# ====== Gaussian linear model =================================================
linear_code = """
data {
    int<lower=0> N; // number of data points 
    vector[N] x; // 
    vector[N] y; // 
}
parameters {
    real alpha; 
    real beta; 
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu <- alpha + beta*x;
}
model {
    y ~ normal(mu, sigma);
}
"""
# Data for Stan
d = np.loadtxt('kilpisjarvi-summer-temp.csv', dtype=np.double, delimiter=';',
               skiprows=1)
x = np.repeat(d[:,0], 4)
y = d[:,1:5].ravel()
N = len(x)
data = dict(N=N, x=x, y=y)
# Compile and fit the model
fit = pystan.stan(model_code=linear_code, data=data)

# Plot
samples = fit.extract(permuted=True)
plt.figure(figsize=(8,10))
plt.subplot(3,1,1)
plt.plot(x,
    np.percentile(samples['mu'], 50, axis=0),
    color='#e41a1c',
    linewidth=1
)
plt.plot(
    x,
    np.asarray(np.percentile(samples['mu'], [5, 95], axis=0)).T,
    color='#e41a1c',
    linestyle='--',
    linewidth=1,
)
plt.scatter(x, y, 5, color='#377eb8')
plt.xlabel('Year')
plt.ylabel('Summer temperature at Kilpisjarvi')
plt.xlim((1952,2013))
plt.subplot(3,1,2)
plt.hist(samples['beta'], 50)
plt.xlabel('beta')
print 'Pr(beta > 0) = {}'.format(np.mean(samples['beta']>0))
plt.subplot(3,1,3)
plt.hist(samples['sigma'], 50)
plt.xlabel('sigma')
plt.tight_layout()
plt.show()


# ====== Gaussian linear model with adjustable priors ==========================
linear_code = """
data {
    int<lower=0> N; // number of data points 
    vector[N] x; // 
    vector[N] y; // 
    real pmualpha; // prior mean for alpha
    real psalpha;  // prior std for alpha
    real pmubeta;  // prior mean for beta
    real psbeta;   // prior std for beta
}
parameters {
    real alpha; 
    real beta; 
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu <- alpha + beta*x;
}
model {
    alpha ~ normal(pmualpha,psalpha);
    beta ~ normal(pmubeta,psbeta);
    y ~ normal(mu, sigma);
}
"""
# Data for Stan
d = np.loadtxt('kilpisjarvi-summer-temp.csv', dtype=np.double, delimiter=';',
               skiprows=1)
x = np.repeat(d[:,0], 4)
y = d[:,1:5].ravel()
N = len(x)
data = dict(
    N = N,
    x = x,
    y = y,
    pmualpha = y.mean(),    # Centered
    psalpha  = (14-4)/6.0,  # avg temp between 4-14
    pmubeta  = 0,           # a priori increase and decrese as likely
    psbeta   = (.1--.1)/6.0 # avg temp probably does not increase more than 1
                            # degree per 10 years
)
# Compile and fit the model
fit = pystan.stan(model_code=linear_code, data=data)

# Plot
samples = fit.extract(permuted=True)
plt.figure(figsize=(8,10))
plt.subplot(3,1,1)
plt.plot(x,
    np.percentile(samples['mu'], 50, axis=0),
    color='#e41a1c',
    linewidth=1
)
plt.plot(
    x,
    np.asarray(np.percentile(samples['mu'], [5, 95], axis=0)).T,
    color='#e41a1c',
    linestyle='--',
    linewidth=1,
)
plt.scatter(x, y, 5, color='#377eb8')
plt.xlabel('Year')
plt.ylabel('Summer temperature at Kilpisjarvi')
plt.xlim((1952,2013))
plt.subplot(3,1,2)
plt.hist(samples['beta'], 50)
plt.xlabel('beta')
print 'Pr(beta > 0) = {}'.format(np.mean(samples['beta']>0))
plt.subplot(3,1,3)
plt.hist(samples['sigma'], 50)
plt.xlabel('sigma')
plt.tight_layout()
plt.show()


# ====== Gaussian linear model with standardized data ==========================
linear_code = """
data {
    int<lower=0> N; // number of data points 
    vector[N] x; // 
    vector[N] y; // 
}
transformed data {
  vector[N] x_std;
  vector[N] y_std;
  x_std <- (x - mean(x)) / sd(x);
  y_std <- (y - mean(y)) / sd(y);
}
parameters {
    real alpha; 
    real beta; 
    real<lower=0> sigma_std;
}
transformed parameters {
    vector[N] mu_std;
    mu_std <- alpha + beta*x_std;
}
model {
  alpha ~ normal(0,1);
  beta ~ normal(0,1);
  y_std ~ normal(mu_std, sigma_std);
}
generated quantities {
    vector[N] mu;
    real<lower=0> sigma;
    mu <- mean(y) + mu_std*sd(y);
    sigma <- sigma_std*sd(y);
}
"""
# Data for Stan
data_path = '../utilities_and_data/kilpisjarvi-summer-temp.csv'
d = np.loadtxt(data_path, dtype=np.double, delimiter=';', skiprows=1)
x = np.repeat(d[:,0], 4)
y = d[:,1:5].ravel()
N = len(x)
data = dict(N = N, x = x, y = y)
# Compile and fit the model
fit = pystan.stan(model_code=linear_code, data=data)

# Plot
samples = fit.extract(permuted=True)
plt.figure(figsize=(8,10))
plt.subplot(3,1,1)
plt.plot(x,
    np.percentile(samples['mu'], 50, axis=0),
    color='#e41a1c',
    linewidth=1
)
plt.plot(
    x,
    np.asarray(np.percentile(samples['mu'], [5, 95], axis=0)).T,
    color='#e41a1c',
    linestyle='--',
    linewidth=1,
)
plt.scatter(x, y, 5, color='#377eb8')
plt.xlabel('Year')
plt.ylabel('Summer temperature at Kilpisjarvi')
plt.xlim((1952,2013))
plt.subplot(3,1,2)
plt.hist(samples['beta'], 50)
plt.xlabel('beta')
print 'Pr(beta > 0) = {}'.format(np.mean(samples['beta']>0))
plt.subplot(3,1,3)
plt.hist(samples['sigma'], 50)
plt.xlabel('sigma')
plt.tight_layout()
plt.show()


# ====== Gaussian linear student-t model =======================================
linear_code = """
data {
    int<lower=0> N; // number of data points 
    vector[N] x; // 
    vector[N] y; // 
}
parameters {
    real alpha; 
    real beta; 
    real<lower=0> sigma;
    real<lower=1,upper=80> nu;
}
transformed parameters {
    vector[N] mu;
    mu <- alpha + beta*x;
}
model {
    nu ~ gamma(2,0.1); // Juarez and Steel (2010)
    y ~ student_t(nu, mu, sigma);
}
"""
# Data for Stan
data_path = '../utilities_and_data/kilpisjarvi-summer-temp.csv'
d = np.loadtxt(data_path, dtype=np.double, delimiter=';', skiprows=1)
x = np.repeat(d[:,0], 4)
y = d[:,1:5].ravel()
N = len(x)
data = dict(N = N, x = x, y = y)
# Compile and fit the model
fit = pystan.stan(model_code=linear_code, data=data)

# Plot
samples = fit.extract(permuted=True)
plt.figure(figsize=(8,12))
plt.subplot(4,1,1)
plt.plot(x,
    np.percentile(samples['mu'], 50, axis=0),
    color='#e41a1c',
    linewidth=1
)
plt.plot(
    x,
    np.asarray(np.percentile(samples['mu'], [5, 95], axis=0)).T,
    color='#e41a1c',
    linestyle='--',
    linewidth=1,
)
plt.scatter(x, y, 5, color='#377eb8')
plt.xlabel('Year')
plt.ylabel('Summer temperature at Kilpisjarvi')
plt.xlim((1952,2013))
plt.subplot(4,1,2)
plt.hist(samples['beta'], 50)
plt.xlabel('beta')
print 'Pr(beta > 0) = {}'.format(np.mean(samples['beta']>0))
plt.subplot(4,1,3)
plt.hist(samples['sigma'], 50)
plt.xlabel('sigma')
plt.subplot(4,1,4)
plt.hist(samples['nu'], 50)
plt.xlabel('nu')
plt.tight_layout()
plt.show()


# ====== Comparison of k groups (ANOVA) ========================================
group_code = """
data {
    int<lower=0> N; // number of data points 
    int<lower=0> K; // number of groups 
    int<lower=1,upper=K> x[N]; // group indicator 
    vector[N] y; // 
}
parameters {
    vector[K] mu;    // group means 
    vector<lower=0>[K] sigma; // group stds 
}
model {
    for (n in 1:N)
      y[n] ~ normal(mu[x[n]], sigma[x[n]]);
}
"""
# Data for Stan
data_path = '../utilities_and_data/kilpisjarvi-summer-temp.csv'
d = np.loadtxt(data_path, dtype=np.double, delimiter=';', skiprows=1)
# Is there difference between different summer months?
x = np.tile(np.arange(1,5), d.shape[0]) # summer months are numbered from 1 to 4
y = d[:,1:5].ravel()
N = len(x)
data = dict(
    N = N,
    K = 4,  # 4 groups
    x = x,  # group indicators
    y = y   # observations
)
# Compile and fit the model
fit = pystan.stan(model_code=group_code, data=data)

# Analyse results
mu = fit.extract(permuted=True)['mu']
# Matrix of probabilities that one mu is larger than other
ps = np.zeros((4,4))
for k1 in range(4):
    for k2 in range(k1+1,4):
        ps[k1,k2] = np.mean(mu[:,k1]>mu[:,k2])
        ps[k2,k1] = 1 - ps[k1,k2]
print "Matrix of probabilities that one mu is larger than other:"
print ps
# Plot
plt.boxplot(mu)
plt.show()


# ====== Hierarchical prior model for comparison of k groups (ANOVA) ===========
# results do not differ much from the previous, because there is only
# few groups and quite much data per group, but this works as an example anyway
hier_code = """
data {
    int<lower=0> N; // number of data points 
    int<lower=0> K; // number of groups 
    int<lower=1,upper=K> x[N]; // group indicator 
    vector[N] y; // 
}
parameters {
    real mu0;             // prior mean 
    real<lower=0> sigma0; // prior std 
    vector[K] mu;         // group means 
    vector<lower=0>[K] sigma; // group stds 
}
model {
    mu0 ~ normal(10,10);      // weakly informative prior 
    sigma0 ~ cauchy(0,4);     // weakly informative prior 
    mu ~ normal(mu0, sigma0); // population prior with unknown parameters
    for (n in 1:N)
      y[n] ~ normal(mu[x[n]], sigma[x[n]]);
}
"""
# Data for Stan
data_path = '../utilities_and_data/kilpisjarvi-summer-temp.csv'
d = np.loadtxt(data_path, dtype=np.double, delimiter=';', skiprows=1)
# Is there difference between different summer months?
x = np.tile(np.arange(1,5), d.shape[0]) # summer months are numbered from 1 to 4
y = d[:,1:5].ravel()
N = len(x)
data = dict(
    N = N,
    K = 4,  # 4 groups
    x = x,  # group indicators
    y = y   # observations
)
# Compile and fit the model
fit = pystan.stan(model_code=hier_code, data=data)

# Analyse results
samples = fit.extract(permuted=True)
print "std(mu0): {}".format(np.std(samples['mu0']))
mu = samples['mu']
# Matrix of probabilities that one mu is larger than other
ps = np.zeros((4,4))
for k1 in range(4):
    for k2 in range(k1+1,4):
        ps[k1,k2] = np.mean(mu[:,k1]>mu[:,k2])
        ps[k2,k1] = 1 - ps[k1,k2]
print "Matrix of probabilities that one mu is larger than other:"
print ps
# Plot
plt.boxplot(mu)
plt.show()

