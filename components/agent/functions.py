import wandb

from components.agent.ppo import Memory
import components.trading_env.env as gym_env
from components.utils.visualize import plot_results


def execute(agent, df, config, plot_path, training):
    train_env = gym_env.AdvancedFuturesTradingEnv(df)
    memory = Memory()
    episodes_stats = []
    full_score = 0
    for episode in range(config['episodes']):
        rewards, actions, prices = agent.predict(memory, train_env, df, config['batch_size'], config['eps_clip'],
                                                 config['c1'], config['c2'], config['c3'], config['gamma'],
                                                 config['lambda_gae'], training)
        score = rewards.sum()
        full_score += score
        episodes_stats.append({score, rewards, actions, prices})
        plot_filepath = f'{plot_path}/{episode + 1}.png'
        plot_results(episodes_stats, plot_filepath)
        wandb.log({'episode': episode, 'score': score, 'rewards': rewards, 'actions': actions, 'prices': prices})

    return full_score
