import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotnine as pn
from mizani.formatters import scientific_format


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    houses = pd.read_csv(filename)
    houses = houses.replace([-np.inf, np.inf], np.nan)
    houses = houses.dropna(axis=0)
    houses["date"] = houses["date"].str.slice(stop=4).astype(np.float64)

    # cleaning outliers
    houses = houses[houses["date"] > 0]
    houses = houses[houses["price"] > 0]
    houses = houses[houses["bedrooms"] > 0]
    houses = houses[houses["bathrooms"] > 0]
    houses = houses[houses["sqft_living"] > 0]
    houses = houses[houses["floors"] > 0]
    houses = houses[(1 <= houses["condition"]) & (houses["condition"] <= 5)]
    houses = houses[(1 <= houses["grade"]) & (houses["grade"] <= 13)]
    houses = houses[houses["yr_built"] > 0]
    houses = houses[(0 <= houses["sqft_lot15"]) & (houses["sqft_lot15"] <= 6000000)]
    # new features
    temp = houses["yr_renovated"].copy(deep=True)
    temp[temp < houses["yr_built"]] = houses["yr_built"][temp < houses["yr_built"]]
    houses["age"] = -1 * (houses["date"] - temp)
    houses["mean_room_size"] = houses["sqft_living"] / (houses["bedrooms"] + houses["bathrooms"])

    # cleaning outliers out of new features

    houses = houses[(50 <= houses["mean_room_size"]) & (houses["mean_room_size"] <= 1200)]

    houses = pd.get_dummies(houses, columns=["zipcode"])
    houses = houses.dropna(axis=0)

    # drop random feature
    houses = houses.drop(columns=["id"])

    price_col = houses["price"]
    houses = houses.drop(columns="price")
    return houses, price_col


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for i in range(X.shape[1]):
        x = X.iloc[:, i].astype(np.float64)
        cov = np.cov(x, y)
        cov_x_y = cov[0, 1]
        sd_x = np.sqrt(cov[0, 0])
        sd_y = np.sqrt(cov[1, 1])
        p = cov_x_y / (sd_x * sd_y)
        g = pn.ggplot(None) + pn.geom_point(pn.aes(X[X.iloc[:, i].name], y), color="green",
                                            alpha=0.25) + pn.theme_classic()
        g = g + pn.ggtitle(
            f"Feature name: {X.iloc[:, i].name}\n"
            f"Pearson Correlation - \n"
            f"between {X.iloc[:, i].name} and price: {p}")
        pn.ggsave(g, f"{output_path}/{X.iloc[:, i].name}_after.png")


def plot_mean_mse(train_X, train_y, test_X, test_y):
    mse_lst_of_mean = []
    mse_lst_of_var = []
    indices = np.arange(train_X.shape[0])
    for p in np.linspace(0.1, 1, 100):
        mse_lst = []
        for i in range(10):
            train_indices = np.random.choice(indices, int(np.ceil(train_X.shape[0] * p)), replace=False)
            train_indices = np.sort(train_indices)
            lr = LinearRegression()
            lr.fit(train_X.iloc[train_indices, :].to_numpy(), (train_y.iloc[train_indices]).to_numpy())

            mse = lr.loss(test_X.to_numpy(), test_y.to_numpy())
            mse_lst.append(mse)
        np_mse_lst = np.array(mse_lst)
        mse_lst_of_mean.append(np.mean(np_mse_lst))
        mse_lst_of_var.append(np.var(np_mse_lst))
    y_min = np.array(mse_lst_of_mean) - 2 * np.sqrt(np.array(mse_lst_of_var))
    y_max = np.array(mse_lst_of_mean) + 2 * np.sqrt(np.array(mse_lst_of_var))
    mse_mean_df = pd.DataFrame(
        data={"percent of train data": list(np.linspace(0.1, 1, 100)), "mean mse": mse_lst_of_mean,
              "lower_bound": list(y_min),
              "upper_bound": list(y_max)})
    g = pn.ggplot(mse_mean_df,
                  pn.aes("percent of train data", "mean mse")) + pn.geom_point() + pn.geom_ribbon(
        pn.aes(ymin=mse_mean_df["lower_bound"], ymax=mse_mean_df["upper_bound"]), fill="red",
        alpha=0.1) + pn.theme_classic() + pn.scale_y_continuous(labels=scientific_format())
    g = g + pn.ggtitle("plot of mean mse vs. percent of the training data")
    # print(g)
    pn.ggsave(g, f"{output_path}/mean_mse.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = "../datasets/house_prices.csv"
    output_path = "../output"
    df, price = load_data(path)

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(df, price, "../output")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, price)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    plot_mean_mse(train_X, train_y, test_X, test_y)
