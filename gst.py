import numpy as np
import matplotlib.pyplot as plt


#def tgauss_smpl(mu, sigma, size=1000):
#    X = np.random.normal(loc=mu, scale=sigma, size=size)
#    X[X<=0] = X[X>0].min()
#    return X


def truncate_sample(distribution, thr=0, s='lower'):
    if s == 'lower':
        def f(size=1000):
            X = distribution(size=size)
            X[X<=thr] = X[X>thr].min()
            return X
    elif s == 'upper':
        def f(size=1000):
            X = distribution(size=size)
            X[X>=thr] = X[X<thr].max()
            return X
    return f


def result_summary(res):#, z=2.0):
    print("mean +- std")
    m, s = np.mean(res), np.std(res)
    print("{:.2e} +- {:.2e}".format(m, s))
    print("----")
    print("95% confidence interval")
    print("[{:.2e}, {:.2e}]".format(np.percentile(res, 2.5), np.percentile(res, 97.5)))
    print("----")
    print("95% confidence Gaussian approx (false)")
    z = 2.0
    sqrN = np.sqrt(len(res))
    print("[{:.2e}, {:.2e}]".format(m-z*s/sqrN, m+z*s/sqrN))


def monte_plot(vs, res, bins=50):
    for i in range(len(vs)):
        plt.figure()
        plt.hist(vs[i], bins=bins)
        plt.title("variable # {}".format(i))

    plt.figure()
    plt.hist(res, bins=bins)
    plt.title("Result")
    plt.show()


def monte_carlo_guesstimate(
        func,
        #mus_sigs,
        distros,
        sample_size=10000):

    #vs = [tgauss_smpl(*a, size=sample_size) for a in mus_sigs]
    vs = [d(size=sample_size) for d in distros]

    res = [func(*a) for a in zip(*vs)]

    result_summary(res)
    monte_plot(vs, res)

    return res


def lognormal(mean, std, **kwargs):
    x   = (std/mean)**2
    mu  = np.log(mean/np.sqrt(1+x))
    sig = np.sqrt(np.log(1+x))
    return np.random.lognormal(mu, sig, **kwargs)


if __name__ == '__main__':
    from functools import partial
    from numpy.random import normal

    """
    Estimate GDP of Russia as p*f*s/60, where p is population, f is fraction of
    income receiving citizens, s is annual income in RUB. "/60" converts RUB to
    USD.
    """
    # truncate_sample is used to prevent negative values
    _ = monte_carlo_guesstimate(
        lambda p,f,s: p*f*s/60,
        [
            truncate_sample(partial(normal, 150*1e6, 10*1e6)),     # distribution of 1st parameter (p)
            truncate_sample(partial(normal, 0.75, 0.10)),          # distribution of 2nd parameter (f)
            truncate_sample(partial(lognormal, 150*1e4, 70*1e4)), # distribution of 3rd parameter (s)
        ]
    )
        #[[150*1e6, 10*1e6], [0.75, 0.10], [150*1e4, 50*1e4]])
