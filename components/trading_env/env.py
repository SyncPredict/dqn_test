import gym
from gym import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """Оптимизированная среда для торговли криптовалютами."""

    def __init__(self, df, window_size=288, stop_loss=0.001, max_loss=0.003, take_profit=0.01):
        super(TradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.stop_loss = stop_loss
        self.max_loss = max_loss
        self.take_profit = take_profit

        self.action_space = spaces.Discrete(3)  # 0 - ничего не делать, 1 - купить, 2 - продать
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 3), dtype=np.float32)

        self.current_step = 0
        self.done = False
        self.position_price = None
        self.max_drawdown = 0
        self.total_profit = 0

    def set_data(self, df):
        """
        Обновляет данные в среде.

        :param df: Новый DataFrame с данными для торговли.
        """
        self.df = df
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.position_price = None
        self.max_drawdown = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        end = self.current_step
        start = end - self.window_size
        obs = self.df.iloc[start:end]
        return obs.values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - self.window_size:
            self.done = True

        current_price = self.df.iloc[self.current_step]
        pnl = 0

        if self.position_price:
            pnl = (current_price - self.position_price) / self.position_price
            self.max_drawdown = min(self.max_drawdown, pnl)

        reward = 0
        if action == 1:  # Купить
            if not self.position_price:
                self.position_price = current_price
        elif action == 2:  # Продать
            if self.position_price:
                reward = pnl - self.max_drawdown
                self.total_profit += pnl
                self.position_price = None
                self.max_drawdown = 0

        if self.position_price:
            if pnl <= -self.stop_loss or self.max_drawdown <= -self.max_loss:
                self.done = True
            elif pnl >= self.take_profit:
                self.done = True

        return self._next_observation(), reward, self.done, current_price

    def render(self, mode='human', close=False):
        profit = self.total_profit * 100
        print(f'Текущий шаг: {self.current_step}, Общая прибыль: {profit:.2f}%')
