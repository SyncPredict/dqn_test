import wandb
from datetime import datetime
import components.data.processor as process_data
import components.trading_env.env as gym_env
from components.agent.agent import Agent
from components.utils.config import Config

def execute(config):
    train_df, test_df = process_data.DataPreprocessor('data.csv').load_data(config['batch_size'])
    train_env = gym_env.TradingEnv(train_df, config['window_size'])
    test_env = gym_env.TradingEnv(test_df, config['window_size'])

    agent = Agent(config)
    train_score = agent.training(train_env, train_df, config)
    agent.save_model('dqn_model.pth')

    test_score = agent.testing(test_env, test_df, config)

    # agent.save_model('dqn_model.pth')
    wandb.log({'Train score': train_score, 'Test score': test_score})


if __name__ == '__main__':
    wandb.init(project='dqn_test')
    config = Config()
    execute(config)
