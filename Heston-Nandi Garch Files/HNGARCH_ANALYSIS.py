import numpy as np
import pandas as pd
import scipy.optimize as spop
import random
import warnings
from scipy.stats import norm

pd.options.mode.copy_on_write = True
warnings.simplefilter(action='ignore', category=FutureWarning)

## Open Files
filename = "Risk_Premium_Analysis_DB"
db = pd.read_excel(filename)

# Rename Columns
rename_dict = {
    'MSCI Global Environment': 'Environment',
    'MSCI Global Alternative Energy': 'Alternative_energy',
    'MSCI Global Energy Efficiency': 'Energy_efficiency',
    'MSCI Global Green Building': 'Green_building',
    'MSCI Global Pollution Prevention': 'Pollution_prevention',
    'MSCI Global Sustainable Water': 'Sustainable_water',
    'MSCI ACWI Net Total Return': 'Total'
}
db.rename(columns=rename_dict, inplace=True)

## Column Date to DateTime
db['Date'] = pd.to_datetime(db['Date'])
db.set_index('Date', inplace=True)
db.index = pd.DatetimeIndex(db.index, freq='B')

## New Dataframe and differentiation
esg_cols = ['Environment', 'Alternative_energy', 'Energy_efficiency',
            'Green_building', 'Pollution_prevention', 'Sustainable_water', 'Total']
df = db[esg_cols].copy()

df_diff = df.diff().add_suffix('_diff')

## Drop first Nan Generated Row
df_diff = df_diff.drop(df_diff.index[0])

## Factor Differentiation
column_names_fact = db.columns[7:12]

for col in column_names_fact:
    db[f'{col}_diff'] = db[col].diff()

db = db.drop(db.index[0])



## Creating the Original HNGARCH and the Explicit-Factors HNGARCH models

class HNGARCH():
    def __init__(self,sigma2_hist,serie,x,nomi, r):
        self.sigma2_hist = sigma2_hist
        self.serie = serie 
        self.x = x   ## Factors
        self.nomi = nomi  ## Factor's Name
        self.r = r  ## Ris-Free Serie
        
        self.loglik_history = []
    
    
    def hngarch_explicit_factors_loglik(self, teta):

        global h, log_lik_value_factor, omega, alpha, beta, gamma, lam, long_var, T, delta

        serie = self.serie
        K = self.x.shape[1]  ## Parameters for the Factors
        r = self.r
        
        ## Sample Lenght
        T = len(serie)
        
        ## Parameters Transformation
        omega = teta[0]**2
        alpha = np.exp(teta[1])
        beta = np.exp(teta[2])
        gamma = teta[3]**2 

        ## Teta Lenght for K parameters of Factor HNGARCH Analysis
        if len(teta) < 4 + K + 1:
            raise ValueError(f"teta doesen't have enough parameters for delta. Set {K+1} parameters for delta.")
    
        # Extract Delta from the Teta Vector
        delta = teta[4:4 + K + 1]
        
        ## Long-run variance and Stationarity Condition
        if (1-beta-alpha*gamma**2) > 0:
            long_var = (omega+alpha)/(1-alpha*gamma**2-beta)
        else: 
            long_var = self.sigma2_hist


        # Vectors of Conditional Variances, log_lik, z, Epsilon, Expected Return and Lambda
        h = np.zeros(T)
        h[0] = long_var

        log_lik = np.zeros(T)
    
        z = np.zeros(T)

        y_hat = np.zeros(T)
        y_hat[0] = serie[0]

        epsilon = np.zeros(T)

        lam = np.zeros(T)
            
        # HNGARCH(1,1) Function ((Explicit Factor Approach))
        for t in range(0, T-1):
                
            h[t+1] = omega + beta*h[t] + alpha*(z[t]-gamma*np.sqrt(h[t]))**2

            if h[t+1]<=0 or np.isnan(h[t+1]) or np.isinf(h[t+1]):
                return 1e10
            
            lam[t+1] = delta[0] + np.dot(self.x[t+1], delta[1:].T)

            y_hat[t+1] = r[t] + lam[t+1]*h[t+1]

            epsilon[t+1] = serie[t+1] - y_hat[t+1]
            
            z[t+1] = (epsilon[t+1])/(np.sqrt(h[t+1]))

            try:
                log_lik[t+1] = -0.5*(np.log(h[t+1]))-0.5*(z[t+1]**2)
            except:
                log_lik[t+1] = 1e50


        # Log-likelihood
        log_lik_value_factor = -np.sum(log_lik)

        return log_lik_value_factor
    
    def results_factors(self):

        K = self.x.shape[1]
        
        ch_convergence_fact = bool(False)

        cont_fact = 0
        
        ## Convergence Loop
        while ch_convergence_fact == False:
        
            ## HNGARCH Parameters Initialization
            alpha_ini_fact = random.gauss(0, 1)
            beta_ini_fact = random.gauss(0, 1)
            gamma_ini_fact = random.gauss(0, 1)

            ### Minimization Function
            initial_params = [self.sigma2_hist, alpha_ini_fact, beta_ini_fact, gamma_ini_fact] + [0] * (K + 1)
            
            res = spop.minimize(self.hngarch_explicit_factors_loglik, initial_params, method='BFGS') ##, tol = 1e-5

            ## Hessian Matrix
            hess_inv_fact=res.hess_inv
            hess_matrix_fact = np.matrix(hess_inv_fact)
            
            ### Standard Errors
            diag_fact = np.diag(hess_matrix_fact)
            stderrors_fact = np.sqrt(diag_fact)
            
            print(f"{'='*50}\n")
            cont_fact = cont_fact + 1
            if diag_fact.sum()!= hess_matrix_fact.shape[0]:
                ch_convergence_fact = True

        print(f'LOOP_HESS_FACT: {cont_fact}')
        
        ### Correlation 
        corr_factor = np.zeros(T)
        for i in range(1, T):
            corr_factor[i] = -((2*gamma*h[i-1])/np.sqrt(4*(gamma**2)*h[i-1]+2)*np.sqrt(h[i]))
            
            
        #### Print Results
        
        print('Inverse Hessian Matrix')
        print('')
        print(hess_inv_fact)
        print('')
        print(f"{'='*50}\n")

        print('Parameters Standard Errors')
        print(stderrors_fact)
        print('')
        print(f"{'='*50}\n")
        
        parametri_f = [omega,alpha,beta,gamma]

        parametriDelta = list(delta)

        ## List of the Parameters
        parametri_tot = parametri_f + parametriDelta

        t_value = []
        p_value = []
        
        for param, err in zip(parametri_tot, stderrors_fact):
            t = param / err
            p_v = 2*(1- norm.cdf(abs(t)))
            
            p_value.append(p_v)
            t_value.append(t)
        
        print(f'P_VALUES == {p_value}')
        print(f'T_VALUES == {t_value}')
        
        ## Estimated Parameters
        print('')
        print('')
        print('HNGARCH(1,1) parameters')
        print('')
        print(f'omega: {omega} ')
        print(f'alpha: {alpha} ')
        print(f'beta: {beta} ')
        print(f'gamma: {gamma} ')

        print('')
        print(f'Intercept: {delta[0]}')
        for i,(delt, nom) in enumerate(zip(range(len(delta)), range(len(self.nomi)))):
            print(f'delta - {self.nomi[nom]}: {delta[delt+1]} ')

        print('')
        print(f"{'='*50}\n")

        print(f'long_run_variance: {long_var} ')
        print(f'log_likelihood: {log_lik_value_factor} ')

        ## Annualized Volatility
        annualized_volatility_factor = np.sqrt(h*252)

        print('')
        print('PERSISTANCE')
        print((beta + alpha*gamma**2))
        print(f"{'='*50}")
        
        
    def hngarch_loglik(self,teta): 

        global h, log_lik_value_factor, omega, alpha, beta, gamma, lam, long_var, T
    
        serie = self.serie
        r = self.r
        
        ## Parameters Transformation
        omega = teta[0]**2
        alpha = np.exp(teta[1])
        beta = np.exp(teta[2])
        gamma = teta[3]**2
        lam = teta[4]

        ## Sample Lenght
        T = len(serie)
        
        ## Long-run variance and Stationarity Condition
        if (1-beta-alpha*gamma**2) > 0:
            long_var = (omega+alpha)/(1-alpha*gamma**2-beta)
        else: 
            long_var = self.sigma2_hist

        # Vectors of Conditional Variances, log_lik, z, Epsilon, Expected Return and Lambda
        h = np.zeros(T)
        h[0] = long_var
    
        log_lik = np.zeros(T)
    
        z = np.zeros(T)

        y_hat = np.zeros(T)
        y_hat[0] = serie[0]

        epsilon = np.zeros(T)
            
        # HNGARCH(1,1) Function
        for t in range(0, T-1):
            
            h[t+1] = omega + beta*h[t] + alpha*(z[t]-gamma*np.sqrt(h[t]))**2

            if h[t+1]<=0 or np.isnan(h[t+1]) or np.isinf(h[t+1]):
                return 1e10

            y_hat[t+1] = r[t] + lam*h[t+1]

            epsilon[t+1] = serie[t+1] - y_hat[t+1]
            
            z[t+1] = (epsilon[t+1])/(np.sqrt(h[t+1]))

            try:
                log_lik[t+1] = -0.5*(np.log(h[t+1]))-0.5*(z[t+1]**2)
            except:
                log_lik[t+1] = 1e50


        # Log-likelihood
        log_lik_value = -np.sum(log_lik)
        
        return log_lik_value
    
    def results(self):
        
        ch_convergence = bool(False)

        cont = 0
        
        ## Convergence Loop
        while ch_convergence == False:
            
            ## HNGARCH Parameters Initialization
            alpha_ini = random.gauss(0, 1)       
            beta_ini = random.gauss(0, 1)           
            gamma_ini = random.gauss(0, 1)            
            lam_ini = random.gauss(0, 1)
            
            res1 = spop.minimize(self.hngarch_loglik, [self.sigma2_hist, alpha_ini, beta_ini, gamma_ini, lam_ini], method='BFGS', tol = 1e-5)

            ## Hessian Matrix
            hess_inv = res1.hess_inv
            hess_matrix = np.matrix(hess_inv)
            
            diag = np.diag(hess_inv)
            
            print(f"{'='*50}\n  ")
            cont = cont+1
            
            if diag.sum()!= hess_matrix.shape[0]:
                ch_convergence = True
        
        print(f'LOOP_HESS: {cont}')


        #### Print Results
        
        print('Matrice Hessiana Inversa')
        print('')
        print(hess_matrix)
        print('')
            
        print('STANDARD ERROR dei parametri')
        stderrors = np.sqrt(diag)
        print(stderrors)
        print(f"{'='*50}\n")
        
        parametri = [omega,alpha,beta,gamma, lam]

        p_value_std = []
        t_value_std = []
        
        for param, err in zip(parametri, stderrors):
            t_std = param / err
            p_v_std = 2*(1-norm.cdf(abs(t_std)))
            
            p_value_std.append(p_v_std)
            t_value_std.append(t_std)
        print(f'P_VALUES == {p_value_std}')
        print(f'T_VALUES == {t_value_std}')
        
        ## Correlation
        corr = np.zeros(T)
        for i in range(1, T):
            corr[i] = -((2*gamma*h[i-1])/np.sqrt(4*(gamma**2)*h[i-1]+2)*np.sqrt(h[i]))
        
        print('')
        print('HNGARCH(1,1) parameters')
        print('')
        print(f'omega: {omega} ')
        print(f'alpha: {alpha} ')
        print(f'beta: {beta} ')
        print(f'gamma: {gamma} ')
        print(f'lambda: {lam} ')

        print('')
        print(f'long_run_variance: {long_var} ')

        ## Annualized Volatility
        annualized_volatility = np.sqrt(h*252)
        
        print('PERSISTANCE')
        print((beta + alpha*gamma**2))
        
        


## Risk-Free Interest Rate 
risk_free = db['Libor USD 1 Month']
r = risk_free/252
r = r.drop(r.index[0])

## Factors
x = db[['Global Corporate IG Option-adjusted Spread_diff', 'World Breakeven Inflation_diff', 'World Real Rate_diff', 'Dollar Index_diff']].values
nomi = ['corporateIG', 'Breakeven Inflation', 'Real Rate', 'Dollar Index']


## Explicit Factors Analysis
results_factors = {}
for i in df_diff:
    serie = df_diff[i]
    sigma_hist = np.std(df_diff[i])**2
    results_factors[f'Result_{i}_factors'] = HNGARCH(sigma_hist, serie, x, nomi , r)
    print(f'{i} Analysis')
    results_factors[f'Result_{i}_factors'].results_factors()
    print(f"{'='*50}")
    print(f"{'='*50}")
    print(f"{'='*50}\n")
   

 
## Taditional HNGARCH Analysis
results = {}
for i in df_diff:
    serie = df_diff[i]
    sigma_hist = np.std(df_diff[i])**2
    results[f'Result_{i}'] = HNGARCH(sigma_hist, serie, x, nomi, r)
    print(f'{i} Analysis')
    results[f'Result_{i}'].results()
    print(f"{'='*50}")
    print(f"{'='*50}")
    print(f"{'='*50}\n")
