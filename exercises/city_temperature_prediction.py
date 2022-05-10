import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotnine as pn


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    daily_temp = pd.read_csv(filename, parse_dates=["Date"])

    # cleaning outliers
    daily_temp["DayOfYear"] = daily_temp["Date"].dt.dayofyear
    daily_temp = daily_temp[daily_temp["Temp"] > -40]

    # drop random feature
    daily_temp = daily_temp.drop(columns=["Date", "Day"])

    # temp_col = daily_temp["Temp"]
    # daily_temp = daily_temp.drop(columns="Temp")
    return daily_temp


def israel_temp_func(daily_temp):
    israel_temp = daily_temp[daily_temp["Country"] == "Israel"].copy(deep=True)
    israel_temp["Year"] = israel_temp["Year"].astype(str)
    g1 = pn.ggplot(israel_temp, pn.aes("DayOfYear", "Temp", color=israel_temp["Year"])) + pn.geom_point(
        alpha=0.5) + pn.theme_classic()
    g1 = g1 + pn.ggtitle("Day of year vs. temperature") + pn.xlab("Day of year") + pn.ylab("temperature")
    pn.ggsave(g1, f"{output_path}/day_temp_scatter.png")
    # print(g1)
    # x^5
    bar = israel_temp.groupby("Month").agg(np.std)
    bar["Month"] = np.arange(1, 13)
    g2 = pn.ggplot() + pn.geom_col(bar, pn.aes("Month", "Temp"), color="blue",
                                   fill="yellow") + pn.theme_classic()
    g2 = g2 + pn.ggtitle("Bar plot of the sd(temperature)") + pn.ylab(
        "sd(temperature)") + pn.scale_x_discrete(
        name="Month", limits=range(1, 13))
    pn.ggsave(g2, f"{output_path}/sd_bar.png")
    # print(g2)


def create_line_plot(daily_temp):
    mean_group = daily_temp.groupby(["Country", "Month"], as_index=False).agg(np.mean)
    sd_group = daily_temp.groupby(["Country", "Month"], as_index=False).agg(np.std)
    mean_group["mean(temperature)"] = mean_group["Temp"]
    mean_group["ymin"] = mean_group["mean(temperature)"] - sd_group["Temp"]
    mean_group["ymax"] = mean_group["mean(temperature)"] + sd_group["Temp"]
    g = pn.ggplot(mean_group,
                  pn.aes("Month", "mean(temperature)",
                         color=mean_group["Country"])) + pn.geom_line() + pn.theme_classic()
    g = g + pn.geom_errorbar(pn.aes(ymin=mean_group["ymin"], ymax=mean_group["ymax"])) + pn.scale_x_discrete(
        name="Month", limits=range(1, 13))
    g = g + pn.ggtitle("Plot of all countries mean(temperature) vs. month")
    pn.ggsave(g, f"{output_path}/line_plot.png")
    # print(g)


def find_best_k_israel(daily_temp):
    israel_temp = daily_temp[daily_temp["Country"] == "Israel"].copy(deep=True)
    temp_col = israel_temp["Temp"]
    israel_temp = israel_temp.drop(columns=["Country", "City", "Month", "Temp"])
    train_X, train_y, test_X, test_y = split_train_test(israel_temp, temp_col)
    mse_lst = []
    for k in np.linspace(1, 10, 10):
        pr = PolynomialFitting(int(k))
        pr.fit(train_X["DayOfYear"].to_numpy(), (train_y.to_numpy()))
        mse = round(pr.loss(test_X["DayOfYear"].to_numpy(), test_y.to_numpy()), 2)
        mse_lst.append(mse)
        print(mse)
    bar = pd.DataFrame({"k": range(1, 11), "mse": mse_lst})
    g = pn.ggplot() + pn.geom_col(bar, pn.aes("k", "mse"), color="black",
                                  fill="pink") + pn.theme_classic()
    g = g + pn.ggtitle("Bar plot of k vs. mse") + pn.scale_x_discrete(name="k", limits=range(1, 11))
    # print(g)
    pn.ggsave(g, f"{output_path}/bar_k_mean.png")
    return np.argmin(np.array(mse_lst)) + 1


def other_countries_fit(k, daily_temp):
    israel_temp = daily_temp[daily_temp["Country"] == "Israel"].copy(deep=True)
    temp_col = israel_temp["Temp"]
    israel_temp = israel_temp.drop(columns=["Country", "City", "Month", "Temp"])
    pr = PolynomialFitting(int(k))
    pr.fit(israel_temp["DayOfYear"].to_numpy(), (temp_col.to_numpy()))
    countries = list(np.unique(daily_temp["Country"]))
    countries.remove("Israel")
    mse_lst = []
    for country in countries:
        country = daily_temp[daily_temp["Country"] == country]
        mse = pr.loss(country["DayOfYear"].to_numpy(), country["Temp"].to_numpy())
        mse_lst.append(mse)
        # print(round(mse, 2))
    bar = pd.DataFrame({"Country": countries, "mse": mse_lst})
    g = pn.ggplot() + pn.geom_col(bar, pn.aes("Country", "mse"), color="black",
                                  fill="red") + pn.theme_classic()
    g = g + pn.ggtitle("Bar plot of country vs. mse")
    # print(g)
    pn.ggsave(g, f"{output_path}/country_mse.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    path = "../datasets/City_Temperature.csv"
    output_path = "../output"
    daily_temp_df = load_data(path)

    # Question 2 - Exploring data for specific country
    israel_temp_func(daily_temp_df)

    # Question 3 - Exploring differences between countries
    create_line_plot(daily_temp_df)

    # Question 4 - Fitting model for different values of `k`
    k_best = find_best_k_israel(daily_temp_df)

    # Question 5 - Evaluating fitted model on different countries
    other_countries_fit(k_best, daily_temp_df)
