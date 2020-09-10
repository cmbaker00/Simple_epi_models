import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd


class SIR_model:
    def __init__(self, beta=2., gamma=1., s0=.9999, i0=.0001):
        self.R0 = beta/gamma
        self.beta = beta
        self.gamma = gamma
        self.s0 = s0
        self.i0 = i0
        self.r0 = 1 - s0 - i0
        self.IC = [s0, i0, self.r0]

        self.t = None
        self.state = None

    def ode_rhs(self, y, t):
        s, i, r = y
        return [-self.beta*i*s,
                self.beta*i*s - self.gamma*i,
                self.gamma*i]

    def solve_system(self, t_end=10):
        tvals = np.linspace(0, t_end, 100)
        return tvals, odeint(self.ode_rhs, self.IC,
                      t=tvals)

    # http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/ComputerExercises/SIR/SIR.html
    #  $t\to\infty$: $R(infinity) = 1-exp(-R_0 * R(infinity))$
    def est_total_infected(self):
        if self.R0 < 1:
            return 0
        def fun(r_inf):
            return r_inf - (1 - np.exp(-self.R0*r_inf))
        return fsolve(fun, np.array(0.5))[0]


class SIR_model_R0(SIR_model):
    def __init__(self, R0 = 2.53):
        super().__init__(beta=R0, gamma=1)

if __name__ == "__main__":
    model = SIR_model(beta=2.5, gamma=1)
    time, vals = model.solve_system(t_end=8)
    # susceptible, infected, recovered = vals
    # new_infections = vals[:,0][0:-1] - vals[:,0][1:]
    plt.plot(time, vals)
    plt.legend(['S', 'I', 'R'])
    plt.xlabel('Time')
    plt.ylabel('Proportion of population')
    plt.savefig('SIR_plot.png')
    plt.show()
    df = pd.DataFrame(np.append(np.reshape(time, (100,-1)), vals, axis=1))
    df.columns = ['Time', 'Suceptible', 'Infected', 'Recovered']
    df.to_csv('SIR_data.csv', index=0, index_label=None)
    # print(model.R0)
    # print(model.est_total_infected())
    # beta_array = np.linspace(0,2.53,50)
    # total_inf = []
    # for beta_value in beta_array:
    #     total_inf.append(SIR_model(beta=beta_value, gamma=1).est_total_infected())
    # plt.plot(beta_array, total_inf)
    # plt.show()
    # print(SIR_model_R0().est_total_infected())