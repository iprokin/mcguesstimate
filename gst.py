import numpy as np
import matplotlib.pyplot as plt

def tgauss_smpl(mu, sigma, size=1000):
    X = np.random.normal(loc=mu, scale=sigma, size=size)
    X[X<=0] = X[X>0].min()
    return X

def monte_plot(vs, res):
    for i in range(len(vs)):
        plt.figure()
        plt.hist(vs[i])
        plt.title("variable # {}".format(i))

    plt.figure()
    plt.hist(res)
    plt.title("Result")
    print("{} +- {}".format(np.mean(res), np.std(res))) 
       
def monte_carlo_guesstimate(
        func,
        mus_sigs,
        sample_size=1000):

    vs = [tgauss_smpl(*a, size=size) for a in mus_sigs]

    res = [func(*a) for a in zip(*vs)]

    monte_plot(vs, res)
    return res

_ = monte_carlo_guesstimate(
    lambda p,f,s: p*f*s,
    [[150*1e6, 10*1e6], [0.66, 0.1], [5*1e5, 2*1e5]])
