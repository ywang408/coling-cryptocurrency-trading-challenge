from datetime import date
from typing import List, Union

from ...utils import RunMode
from .base import BasePromptConstructor

# memory layer + id
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
warmup_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
test_investment_info_prefix = "The ticker of the cryptocurrency to be analyzed is {symbol} and the current date is {cur_date}"

# sentiment  + momentum explanation
sentiment_explanation = """In investment decision-making, analyzing financial sentiment is pivotal, offering insights into market perceptions and forecasting potential market trends. Sentiment analysis divides market opinions into three categories: positive, negative, and neutral. A positive sentiment signals optimism about future prospects, often leading to increased buying activity, while negative sentiment reflects pessimism, 
                        likely causing selling pressures. Neutral sentiment indicates either uncertainty or a balanced view, suggesting that investors are neither overly bullish nor bearish. Leveraging these sentiment indicators enables investors and analysts to fine-tune their strategies to better match the prevailing market atmosphere.
                        Additionally, news about competitors can significantly impact a company's cryptocurrency price. For example, if a competitor unveils a groundbreaking product, it may lead to a decline in the cryptocurrency prices of other companies within the same industry as investors anticipate potential market share losses.
                        """
momentum_explanation = """The information below provides a summary of cryptocurrency price fluctuations over the previous few days, which is the "Momentum" of a cryptocurrency.
        It reflects the trend of a cryptocurrency.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# summary
warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
test_reason = "Given the information of text and the summary of the cryptocurrency price movement. Please explain the reason why you make the investment decision."

# action
test_action_choice = "Given the information, please make an investment decision: buy the cryptocurrency, sell, and hold the cryptocurrency"

# final prompt
warmup_final_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
        Your should provide a summary information and the id of the information to support your summary."""
test_final_prompt = """Given the information, can you make an investment decision? Just summarize the reason of the decision.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical cryptocurrency price.
    When momentum or cumulative return is positive, you are a risk-seeking investor. 
    In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.
    please consider how much shares of the cryptocurrency the investor holds now.
    You should provide exactly one of the following investment decisions: buy or sell.
    When it is very hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.
    You also need to provide the ids of the information to support your decision."""


# prompt construction
def _add_momentum_info(momentum: int, investment_info: str) -> str:
    if momentum == -1:
        investment_info += (
            "The cumulative return of past 3 days for this cryptocurrency is negative."
        )
    elif momentum == 0:
        investment_info += (
            "The cumulative return of past 3 days for this cryptocurrency is zero."
        )
    elif momentum == 1:
        investment_info += (
            "The cumulative return of past 3 days for this cryptocurrency is positive."
        )
    return investment_info


class VLLMPromptConstructor(BasePromptConstructor):
    @staticmethod
    def __call__(
        cur_date: date,
        symbol: str,
        run_mode: RunMode,
        future_record: Union[float, None],
        short_memory: Union[List[str], None],
        short_memory_id: Union[List[int], None],
        mid_memory: Union[List[str], None],
        mid_memory_id: Union[List[int], None],
        long_memory: Union[List[str], None],
        long_memory_id: Union[List[int], None],
        reflection_memory: Union[List[str], None],
        reflection_memory_id: Union[List[int], None],
        momentum: Union[int, None] = None,
    ) -> str:
        # investment info + memories
        if run_mode == RunMode.WARMUP:
            investment_info = warmup_investment_info_prefix.format(
                symbol=symbol, cur_date=cur_date, future_record=future_record
            )
        else:
            investment_info = test_investment_info_prefix.format(
                symbol=symbol, cur_date=cur_date
            )
        if short_memory and short_memory_id:
            investment_info += "The short-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
            )
            investment_info += sentiment_explanation
            investment_info += "\n\n"
        if mid_memory and mid_memory_id:
            investment_info += "The mid-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
            )
            investment_info += "\n\n"
        if long_memory and long_memory_id:
            investment_info += "The long-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
            )
            investment_info += "\n\n"
        if reflection_memory and reflection_memory_id:
            investment_info += "The reflection-term information:\n"
            investment_info += "\n".join(
                [
                    f"{i[0]}. {i[1]}"
                    for i in zip(reflection_memory_id, reflection_memory)
                ]
            )
        if momentum:
            investment_info += momentum_explanation
            investment_info = _add_momentum_info(momentum, investment_info)

        if run_mode == RunMode.WARMUP:
            return investment_info + warmup_final_prompt
        else:
            return investment_info + test_final_prompt
