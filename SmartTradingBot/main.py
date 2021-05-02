from typing import Optional, Tuple

import numpy as np
import tqdm
from docopt import docopt
from pandas.core.frame import DataFrame

from SmartTradingBot import agent, utils


def run_bot() -> None:
    """Evaluate a trained agent on given data and test performance."""
    return


def train_bot(
    agent: agent.DQNAgent,
    data: DataFrame,
    episode: int,
    n_episodes: int = 50,
    window_size: int = 10,
) -> Tuple:
    """Train the given agent with the provided data."""
    total_reward = 0
    losses = []
    data_length = len(data) - 1

    purchased = []
    avg_loss = []  # type:ignore

    data = utils.normalised_difference(data)
    state = utils.padded_window(data, timestep=0, window_size=window_size)
    # print(state)
    # print(type(state))

    for t in tqdm.tqdm(
        range(data_length), desc="Episode {}/{}".format(episode, n_episodes)
    ):
        reward = 0
        next_state = utils.padded_window(data, timestep=t + 1, window_size=10)
        action = agent.act(state)

        # BUY
        if action == 1:
            purchased.append(data[t])

        # SELL
        elif action == 2 and len(purchased) > 0:
            bought_price = purchased.pop(0)
            delta = data[t] - bought_price
            reward = delta  # max(delta, 0)
            total_reward += delta

        # HODL ;)
        else:
            pass

        done = t == data_length - 1
        loss = agent.step(state, action, reward, next_state, done)
        if loss is not False:
            losses.append(loss)
        state = next_state

    # if episode % 10 == 0:
    #     agent.save(episode)

    return (episode, n_episodes, total_reward, np.mean(np.array(avg_loss)))


def main(
    asset: str,
    start_date: str,
    end_date: str,
    window_size: int = 10,
    batch_size: int = 64,
    n_episodes: int = 50,
    model_name: Optional[str] = None,
    differences: int = 1,
    normalise: bool = True,
) -> None:
    train, test = utils.get_data(
        asset,
        start_date=start_date,
        end_date=end_date,
        normalise=normalise,
        differences=differences,
    )

    trading_agent = agent.DQNAgent(
        state_dim=10,  # 10 days data is one "state1"/feature
        action_dim=3,  # [Hold,Buy,Sell] = [0,1,2]
        hidden_layer_sizes=[128, 256, 256, 128],
        buffer_size=1000,
        batch_size=batch_size,  # 32
        discount=0.99,
        learning_rate=1e-3,
        learning_freq=4,
    )

    results = []
    for episode in range(1, n_episodes):
        x = train_bot(
            agent=trading_agent,
            data=train,
            episode=episode,
            n_episodes=n_episodes,
            window_size=window_size,
        )
        results.append(x)


if __name__ == "__main__":
    args = docopt(__doc__)

    asset = args["<asset>"]
    start_date = args["--start_date"]
    end_date = args["--end_date"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    n_episodes = int(args["--num-episodes"])
    model_name = args["--model-name"]
    differences = args["--differences"]
    normalise = bool(args("--normalise"))

    main(
        asset,
        start_date,
        end_date,
        window_size,
        batch_size,
        n_episodes,
        model_name,
        differences,
        normalise,
    )
