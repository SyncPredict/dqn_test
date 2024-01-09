import os

from matplotlib import pyplot as plt


def print_progress(batch, num_batches, reward):
    progress = (batch + 1) / num_batches
    bar_length = 40
    bar = '#' * int(progress * bar_length) + '-' * (bar_length - int(progress * bar_length))
    string = f'\r[{bar}] Batch {batch + 1}/{num_batches} ({progress * 100:.2f}%) Reward: {reward:.2f}'
    end = '\n' if batch + 1 == num_batches else ''
    print(string, end=end)


def plot_results(batch_result, filepath):
    rewards = batch_result['rewards']
    actions = batch_result['actions']
    prices = batch_result['prices']
    score = batch_result['score']

    plt.figure(figsize=(20, 10))

    # График наград за шаг
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Rewards per Step')
    plt.title(f'Rewards per Step - Total Score: {score}')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()

    # График действий агента и цен на Bitcoin
    plt.subplot(2, 1, 2)
    plt.plot(prices, label='Bitcoin Price', color='black')
    for i, action in enumerate(actions):
        if action == 1:  # Купить
            plt.scatter(i, prices[i], color='green', label='Buy' if i == 0 else "")
        elif action == 2:  # Продать
            plt.scatter(i, prices[i], color='red', label='Sell' if i == 0 else "")

    plt.title('Bitcoin Price and Agent Actions')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


    plt.savefig(filepath)
    plt.savefig(filepath)
    plt.close()
