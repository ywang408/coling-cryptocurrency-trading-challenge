import json
import os
from datetime import datetime
from typing import Tuple, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import FinMemAgent
from .utils import TaskType


def input_data_restructure(
    start_date: str, end_date: str, data_path: str
) -> Tuple[List[datetime], pd.DataFrame]:
    with open(data_path, "r") as file:
        data = json.load(file)

    crypto_dates = []
    crypto_prices = []
    for date, contents in data.items():
        if (
            (contents is not None)
            and ("prices" in list(contents.keys()))
            and (contents["prices"] is not None)
        ):
            crypto_prices.append(contents["prices"])
            crypto_date = datetime.strptime(date, "%Y-%m-%d").date()
            crypto_dates.append(crypto_date)
    # Create price DataFrame
    crypto_df = pd.DataFrame({"Date": crypto_dates, "Adj Close": crypto_prices})
    crypto_df_full = crypto_df.sort_values("Date")
    crypto_df = crypto_df_full[
        (crypto_df_full["Date"] >= datetime.strptime(start_date, "%Y-%m-%d").date())
        & (crypto_df_full["Date"] <= datetime.strptime(end_date, "%Y-%m-%d").date())
    ]
    crypto_df = crypto_df.reset_index(drop=True)
    full_dates_lst = crypto_df["Date"].tolist()

    return full_dates_lst, crypto_df


def reframe_data_files(
    start_date: str,
    end_date: str,
    full_dates_lst: List[datetime],
    ticker: str,
    result_path: str,
) -> pd.DataFrame:
    # Load agent from checkpoint
    action_path = os.path.join(result_path, "agent")
    agent = FinMemAgent.load_checkpoint(
        path=action_path, task_type=TaskType.SingleAsset
    )

    # Create and preprocess DataFrame
    action_df = pd.DataFrame(agent.portfolio.get_action_record())
    action_df.drop(columns="price", inplace=True)  # Drop price column
    action_df.rename(columns={"position": "direction"}, inplace=True)
    action_df["date"] = pd.to_datetime(action_df["date"])
    action_df["date"] = action_df["date"].dt.date
    # Filter data within date range
    mask = (action_df["date"] >= pd.to_datetime(start_date).date()) & (
        action_df["date"] <= pd.to_datetime(end_date).date()
    )
    filtered_df = action_df[mask]

    # Identify missed dates
    missed_dates = [
        date for date in full_dates_lst if date not in filtered_df["date"].tolist()
    ]
    missed_data_df = pd.DataFrame(
        {
            "date": missed_dates,
            "symbol": [ticker] * len(missed_dates),
            "direction": [0] * len(missed_dates),
        }
    )

    return (
        pd.concat([filtered_df, missed_data_df])
        .sort_values(by="date")
        .reset_index(drop=True)
    )


def reward_list(price: List[float], actions: List[float]) -> List[float]:
    """
    Calculates the cumulative reward for a given list of prices and actions.

    Parameters:
        price (list): List of stock prices.
        actions (list): List of actions taken on the stock.

    Returns:
        list: List of cumulative rewards calculated from the prices and actions.
    """
    reward = 0
    reward_list = [0.0]
    for i in range(len(price) - 1):
        reward += actions[i] * np.log(price[i + 1] / price[i])
        reward_list.append(reward)
    return reward_list


def standard_deviation(reward_list):
    """
    float: Standard deviation of the rewards.
    """
    mean = sum(reward_list) / len(reward_list)
    variance = sum((r - mean) ** 2 for r in reward_list) / (len(reward_list) - 1)
    return variance**0.5


def daily_reward(price_list, actions_list):
    reward = []
    for i in range(len(price_list) - 1):
        r = actions_list[i] * np.log(price_list[i + 1] / price_list[i])
        reward.append(r)
    return reward


def total_reward(price_list, actions_list):
    return sum(
        actions_list[i] * np.log(price_list[i + 1] / price_list[i])
        for i in range(len(price_list) - 1)
    )


def annualized_volatility(daily_std_dev, trading_days=365):
    return daily_std_dev * (trading_days**0.5)


def calculate_sharpe_ratio(Rp, Rf, sigma_p, price_list):
    if sigma_p == 0:
        raise ValueError("Standard deviation cannot be zero.")
    Rp = Rp / (len(price_list) / 252)
    return (Rp - Rf) / sigma_p


def calculate_max_drawdown(daily_returns):
    """
    Calculate the maximum drawdown of a portfolio.

    Parameters:
    daily_returns (list): List of daily returns.

    Returns:
    float: Maximum drawdown.
    """
    cumulative_returns = [1]
    for r in daily_returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))

    peak = cumulative_returns[0]
    max_drawdown = 0

    for r in cumulative_returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def calculate_metrics(price_list, actions_list):
    """
    Calculate various financial metrics based on price and actions.

    Parameters:
    price (list): List of daily prices.
    actions (list): List of actions taken.

    Returns:
    tuple: A tuple containing calculated metrics (standard deviation, annualized volatility, cumulative return, Sharpe ratio, max drawdown).
    """
    daily_rewards = daily_reward(price_list, actions_list)
    std_dev_r = standard_deviation(daily_rewards)
    ann_vol = annualized_volatility(std_dev_r)
    cum_return = total_reward(price_list, actions_list)
    sharpe_ratio = calculate_sharpe_ratio(cum_return, 0, ann_vol, price_list)
    return cum_return, sharpe_ratio


def metrics_summary(
    ticker: str,
    price_list: List[float],
    actions_list: List[float],
    output_path: str,
) -> None:
    """
    Main function to calculate metrics and save results to a CSV file.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    start (str): Start date for analysis.
    end (str): End date for analysis.
    df_paths (dict): Dictionary of file paths for different models.
    col_names (list): List containing the names of the date and action columns.
    save_path (str): Path to save the results CSV file.
    """
    metrics = ["Cumulative Return", "Sharpe Ratio"]
    results = {"Buy & Hold": calculate_metrics(price_list, [1] * len(price_list))}
    results[ticker] = calculate_metrics(price_list, actions_list)
    df_results = pd.DataFrame(results, index=metrics)
    df_results.rename(columns={ticker: f"{ticker}"}, inplace=True)
    save_path = os.path.join(output_path, f"{ticker}_metrics.csv")
    df_results.to_csv(save_path)
    print(df_results)
    print("*-*" * 30)


def plot_cumulative_returns(
    dates: List[datetime],
    return_lists: List[List[float]],
    labels: List[str],
    ticker: str,
    output_path: str,
) -> None:
    """
    Plots cumulative returns using the provided data.

    Parameters:
        dates (list): List of dates for the x-axis.
        return_lists (list of lists): List of return lists to be plotted.
        labels (list): List of labels for each return series.
        ticker (str): Stock ticker symbol for the title.
        file_path (str): Path to save the plot.
        Start_Date (bool, optional): Whether to start x-ticks from a specific date. Defaults to True.

    Returns:
        None: The function generates and displays a plot.
    """
    colors = ["tab:blue", "tab:orange"]
    linestyles = ["-.", "-"]
    linewidths = [2.5, 3]
    alphas = [0.66, 1]  # match the length of len(df_dict)+1
    fig, ax = plt.subplots(figsize=(18, 10))

    # Loop through the return lists and plot each one
    for returns, label, color, linestyle, alpha, linewidth in zip(
        return_lists, labels, colors, linestyles, alphas, linewidths
    ):
        ax.plot(
            dates,  # type: ignore
            returns,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
        )

    # Set the labels and title
    ax.set_xlabel("Date", fontsize=28)
    ax.set_ylabel("Cumulative Return", fontsize=28)
    plt.title(f"{ticker}", fontsize=35)

    # Customize the legend
    ax.legend(fontsize=22, frameon=True)

    # Customize the grid
    ax.grid(True)

    # Customize the tick labels on both axes
    ax.tick_params(axis="x", labelsize=22, width=2, rotation=45)  # Rotate x-axis labels
    ax.tick_params(axis="y", labelsize=22, width=2)  # y-axis labels

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    plt.tight_layout()
    output_path = os.path.join(
        output_path, f"{ticker}_cr_BH_FinMem_compare.png"
    )
    plt.savefig(output_path, format="png", dpi=400)


def output_metrics_summary(
    start_date: str,
    end_date: str,
    ticker: str,
    output_path: str,
    data_path: str,
    result_path: str,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    
    full_dates_lst, yahoo_df = input_data_restructure(
        start_date=start_date, end_date=end_date, data_path=data_path
    )
    ticker_stock_price_lst = yahoo_df["Adj Close"].tolist()

    data_df_combined_sorted = reframe_data_files(
        start_date=start_date,
        end_date=end_date,
        result_path=result_path,
        full_dates_lst=full_dates_lst,
        ticker=ticker,
    )

    ticker_actions_lst = data_df_combined_sorted["direction"].tolist()
    labels = ["B_H", f"FinMem-{ticker}"]
    # Calculate rewards
    B_H = [1.0] * len(ticker_stock_price_lst)
    B_H_rewards = reward_list(ticker_stock_price_lst, B_H)
    return_lists = [B_H_rewards]
    ### calculate return list for finmem
    FinMem_rewards = reward_list(ticker_stock_price_lst, ticker_actions_lst)
    return_lists.append(FinMem_rewards)

    metrics_summary(
        ticker=ticker,
        price_list=ticker_stock_price_lst,
        actions_list=ticker_actions_lst,
        output_path=output_path,
    )

    plot_cumulative_returns(
        dates=pd.to_datetime(full_dates_lst).tolist(),  # type: ignore
        return_lists=return_lists,
        labels=labels,
        ticker=ticker,
        output_path=output_path,
    )
