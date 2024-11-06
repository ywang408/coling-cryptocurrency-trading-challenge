import json
import pandas as pd

def format_investment_decision_text(asset_key, json_file='info.json', output_csv='investment_questions.csv'):
    with open(json_file, 'r') as file:
        data = json.load(file)

    formatted_data = {}

    # Loop through each date in the JSON file
    for date, date_data in data.items():
        btc_data = date_data.get(asset_key, {})

        # Extract relevant information
        short_memories_content = btc_data.get("short_memories", {}).get("content", [])
        mid_memories_content = btc_data.get("mid_memories", {}).get("content", [])
        long_memories_content = btc_data.get("long_memories", {}).get("content", [])
        reflection_memories_content = btc_data.get("reflection_memories", {}).get("content", [])
        momentum = btc_data.get("market_data", {}).get("momentum")

        short_memories_text = "\n".join(short_memories_content) if short_memories_content else "No short-term information available."
        mid_memories_text = "\n".join(mid_memories_content) if mid_memories_content else "No mid-term information available."
        long_memories_text = "\n".join(long_memories_content) if long_memories_content else "No long-term information available."
        reflection_memories_text = "\n".join(reflection_memories_content) if reflection_memories_content else "No reflection-term information available."

        # Construct the complete text with preset sentences and extracted data for the current date
        formatted_text = (
            f"Date: {date}\n\n"
            "Given the information, can you make an investment decision?\n"
            "please consider the mid-term information, the long-term information, the reflection-term information only when they are available. "
            "If there no such information, directly ignore the impact for absence such information.\n"
            "please consider the available short-term information and sentiment associated with them.\n"
            "please consider the momentum of the historical cryptocurrency price.\n"
            "When momentum or cumulative return is positive, you are a risk-seeking investor.\n"
            "In particular, you should choose to 'buy' when the overall sentiment is positive or momentum/cumulative return is positive.\n"
            "please consider how much shares of the cryptocurrency the investor holds now.\n"
            "You should provide exactly one of the following investment decisions: buy or sell.\n"
            "When it is very hard to make a 'buy'-or-'sell' decision, then you could go with 'hold' option.\n\n"
            f"The short-term information:\n{short_memories_text}\n"
            f"The mid-term information:\n{mid_memories_text}\n"
            f"The long-term information:\n{long_memories_text}\n"
        )
        
        # Add momentum information only if it's not null
        if momentum is not None:
            formatted_text += (
                f"Momentum: The information below provides a summary of cryptocurrency price fluctuations over the previous few days, "
                f"which is the 'Momentum' of a cryptocurrency. It reflects the trend of a cryptocurrency. Momentum is based on the idea "
                f"that securities that have performed well in the past will continue to perform well, and conversely, securities that "
                f"have performed poorly will continue to perform poorly.\n{momentum}\n"
            )

        formatted_data[date] = formatted_text

    # Convert the dictionary to a DataFrame with 'date' as the index and 'question' as the column
    df = pd.DataFrame.from_dict(formatted_data, orient='index', columns=['question'])
    df.index.name = 'date'

    df.to_csv(output_csv)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    format_investment_decision_text("BTC-USD", "test_queried_infos.json", "BTC_questions.csv")
