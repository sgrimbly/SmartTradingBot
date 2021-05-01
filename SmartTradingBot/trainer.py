import tqdm
import numpy as np

from SmartTradingBot import utils

def train_bot(agent, data, episode, n_episodes=50):
    total_reward = 0
    losses = []
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    data = utils.normalised_difference(data)
    state = utils.padded_window(data, timestep=0, window_size=10)
    #print(state)
    #print(type(state))

    for t in tqdm.tqdm(range(data_length), desc='Episode {}/{}'.format(episode, n_episodes)):        
        reward = 0
        next_state = utils.padded_window(data, timestep=t+1, window_size=10)
        action = agent.act(state)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_reward += delta

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        loss = agent.step(state, action, reward, next_state, done)
        if loss is not False:
            losses.append(loss)
        state = next_state    

    #if episode % 10 == 0:
    #     agent.save(episode)

    return (episode, n_episodes, total_reward, np.mean(np.array(avg_loss)))