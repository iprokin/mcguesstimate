import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import numpy.random as npr


def file_to_str(filepath):
    with open(filepath, "r") as myfile:
        data = myfile.read()
    return data


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


def result_summary(res):
    print("median +- mad")
    m = np.median(res)
    s = np.median(np.abs(res-m))
    print("{:.2e} +- {:.2e}".format(m, s))
    print("----")
    print("95% of values are inside this interval:")
    print("[{:.2e}, {:.2e}]".format(np.percentile(res, 2.5), np.percentile(res, 97.5)))


def monte_plot(vs, res, bins=50, var_names=None, result_name=None):
    for i in range(len(vs)):
        f = plt.figure(figsize=(3,3))
        plt.hist(vs[i], bins=bins)
        if var_names is None:
            vn = "variable # {}".format(i)
        else:
            vn = var_names[i]
        plt.title(vn)
        f.savefig('./'+vn.replace(' ','_')+'.png', bbox_inches='tight', pad_inches=0.1)

    f = plt.figure(figsize=(3,3))
    plt.hist(res, bins=bins)
    if result_name is None:
        rn = "Result"
    else:
        rn = "Result ({})".format(result_name)
    plt.title(rn)
    f.savefig('./'+rn.replace(' ','_')+'.png', bbox_inches='tight', pad_inches=0.1)

    plt.show()


def monte_carlo_guesstimate(
        func,
        distros,
        sample_size=10000,
        var_names=None,
        result_name=None):

    vs = [d(size=sample_size) for d in distros]

    res = [func(*a) for a in zip(*vs)]

    result_summary(res)
    monte_plot(vs, res, var_names=var_names, result_name=result_name)

    return res


def lognormal_mean_std(mean, std, **kwargs):
    x   = (std/mean)**2
    mu  = np.log(mean/np.sqrt(1+x))
    sig = np.sqrt(np.log(1+x))
    return np.random.lognormal(mu, sig, **kwargs)


default_model={
    'formula':
        lambda p, f, s: p*f*s/60,
    'distributions_for_each_parameter': [ # same order as in formula
        truncate_sample(partial(npr.normal, 150*1e6, 10*1e6)),     # distribution of 1st parameter (p)
        truncate_sample(partial(npr.normal, 0.75, 0.10)),          # distribution of 2nd parameter (f)
        truncate_sample(partial(lognormal_mean_std, 150*1e4, 80*1e4)),  # distribution of 3rd parameter (s)
    ],
    'var_names': [
        'Population',
        'Fraction of income receiving people',
        'Annual income'
    ],
    'result_name': 'GDP',
    'description': """
    Estimate GDP of Russia as p*f*s/60, where p is population, f is fraction of
    income receiving citizens, s is annual income in RUB. "/60" converts RUB to
    USD.
    """
}


def mcgwrapper(model):
    print(model['description'])
    print('-'*len(model['description'].split('\n')[1]))
    _ = monte_carlo_guesstimate(
        model['formula'],
        model['distributions_for_each_parameter'],
        var_names=model['var_names'],
        result_name=model['result_name']
    )


if __name__ == '__main__':
    import sys

    model = default_model
    if len(sys.argv)>1:
        if sys.argv[1] in ['--help', '-h']:
            print("Pass path to a file with model as argument. See example file.")
        else:
            exec(file_to_str(sys.argv[1]))
    mcgwrapper(model)
