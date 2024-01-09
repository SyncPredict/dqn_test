import os
import numpy as np
import torch
import wandb
from components.utils.visualize import print_progress, plot_results
from components.agent.dqn import DQNAgent  # Импорт класса DQNAgent


class Agent:
    def __init__(self, config):
        self.agent = DQNAgent(
            input_dim=config['input_dim'],
            hidden_size=config['hidden_size'],
            num_actions=3,  # Купить, Продать, Пасс
            lr=config['lr'],
            gamma=config['gamma'],
            epsilon=config['epsilon'],
            epsilon_decay=config['epsilon_decay'],
            min_epsilon=config['min_epsilon'],
            memory_size=config['memory_size']
        )
        self.batch_size = config['batch_size']

    def training(self, env, df, config):
        return self._execute_episodes(env, df, config['train_episodes'], True)

    def testing(self, env, df, config):
        return self._execute_episodes(env, df, config['test_episodes'], False)

    def _execute_episodes(self, env, df, episodes, training):
        episodes_stats = []
        full_score = 0
        hidden_state = None

        for episode in range(episodes):
            rewards, actions, prices = self.act(env, df, hidden_state)
            score = sum(rewards)
            full_score += score
            batch_result = {'score': score, 'rewards': rewards, 'actions': actions, 'prices': prices}
            if not training:
                plot_results(batch_result, f'plots/episode_{episode}')
            episodes_stats.append(batch_result)
            wandb.log({'Episode score': score})
            self.save_model(f'model_episode_{episode}.pth')

            if training and episode % self.batch_size == 0:
                self.agent.optimize(self.batch_size)

        return full_score

    def act(self, env, df, hidden_state):
        num_batches = len(df) // self.batch_size
        actions, prices, rewards = [], [], []

        for batch in range(num_batches):
            start_step, end_step = batch * self.batch_size, min((batch + 1) * self.batch_size, len(df))
            data, state = df[start_step:end_step], env.set_data(df[start_step:end_step])
            batch_rewards, batch_actions, batch_prices, hidden_state = self._process_batch(state, data, env,
                                                                                           hidden_state)
            actions.extend(batch_actions)
            rewards.extend(batch_rewards)
            prices.extend(batch_prices)

            print_progress(batch, num_batches, sum(rewards[-self.batch_size:]))
            wandb.log({'Batch score': sum(batch_rewards)})

        return rewards, np.array(actions), prices

    def _process_batch(self, state, data, env, hidden_state):
        actions, prices, rewards = [], [], []
        for _ in data:
            action, hidden_state = self.agent.select_action(state, hidden_state)
            next_state, reward, done, price = env.step(action)
            self.agent.store_transition(state, action, next_state, reward, done)
            state = next_state

            actions.append(action)
            rewards.append(reward)
            prices.append(price)
            wandb.log({'Step score': reward})

            if done:
                break

        return rewards, actions, prices, hidden_state

    def save_model(self, filename):
        torch.save(self.agent.model.state_dict(), filename)

    def load_model(self, filename):
        self.agent.model.load_state_dict(torch.load(filename))
