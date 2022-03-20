from plotnine import geom_point, theme_classic, ggtitle

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio
# pio.templates.default = "simple_white"
from plotnine import *
import pandas as pd


def test_univariate_gaussian():
    """
    Question 1 - 3 answers
    """
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    rand_data = np.random.normal(mu, sigma, size=1000)
    ug = UnivariateGaussian()
    ug.fit(rand_data)
    print(ug.mu_, ug.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ranges = np.arange(10, 1001, 10)
    mu_lst = []
    for end_idx in ranges:
        ug.fit(rand_data[:end_idx])
        mu_lst.append(ug.mu_)
    mu_arr = np.array(mu_lst)
    df = pd.DataFrame({"Number of samples": ranges, "abs(estimated-mu - true-mu)": np.abs(mu_arr - mu)})
    g2 = ggplot(df, aes("Number of samples", "abs(estimated-mu - true-mu)")) + geom_point(
        color="gray") + theme_classic() + ggtitle(
        "The influence of samples number of the estimation of mu")
    ggsave(g2, "q2_plot.png", verbose=False)
    print(g2)

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_rand_data = np.sort(rand_data)
    pdf_rand_data = ug.pdf(sorted_rand_data)
    df = pd.DataFrame({"Values": sorted_rand_data, "PDF": pdf_rand_data})
    g3 = ggplot(df, aes("Values", "PDF")) + geom_point(
        color="gray") + theme_classic() + ggtitle(
        "The empirical PDF of the data")
    ggsave(g3, "q3_plot.png", verbose=False)
    print(g3)


def test_multivariate_gaussian():
    """
    Question 4 - 6 answers
    """
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.zeros((4, 4))
    sigma[0, :] = np.array([1, 0.2, 0, 0.5])
    sigma[1, 0:2] = np.array([0.2, 2])
    sigma[2, 2] = 1
    sigma[3, :] = np.array([0.5, 0, 0, 1])

    rand_data = np.random.multivariate_normal(mu, sigma, size=1000)
    mg = MultivariateGaussian()
    mg.fit(rand_data)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)
    f1, f3 = np.meshgrid(space, space)
    f1, f3 = f1.flatten(), f3.flatten()
    zero_arr = np.zeros(f1.size)
    mu = np.array((f1, zero_arr, f3, zero_arr)).T
    f = lambda x: mg.log_likelihood(x, sigma, rand_data)
    values = np.apply_along_axis(f, 1, mu)

    df = pd.DataFrame({"f1": f1, "f3": f3, "LL value": values})
    g5 = ggplot(df, aes("f1", "f3", fill=df["LL value"])) + geom_tile() + ggtitle(
        "Heatmap of LL value with different f1 and f3") + theme_classic()
    ggsave(g5, "q5_heatmap.png", verbose=False)
    print(g5)

    # Question 6 - Maximum likelihood
    max_idx = np.argmax(values)
    max_f1 = f1[max_idx]
    max_f3 = f3[max_idx]
    print(max_f1, max_f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
