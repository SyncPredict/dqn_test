from dotenv import load_dotenv, dotenv_values

class Config:
    def __init__(self):
        load_dotenv()
        self.env_vars = dotenv_values(".env")

        # Здесь определяются основные параметры обучения
        self.train_episodes = self.get_env_variable("train_episodes", int)
        self.test_episodes = self.get_env_variable("test_episodes", int)
        self.input_dim = self.get_env_variable("input_dim", int)
        self.lr = self.get_env_variable("lr", float)
        self.hidden_size = self.get_env_variable("hidden_size", int)
        self.window_size = self.get_env_variable("window_size", int)
        self.batch_size = self.get_env_variable("batch_size", int)
        self.gamma = self.get_env_variable("gamma", float)
        self.epsilon = self.get_env_variable("epsilon", float)
        self.epsilon_decay = self.get_env_variable("epsilon_decay", float)
        self.min_epsilon = self.get_env_variable("min_epsilon", float)
        self.memory_size = self.get_env_variable("memory_size", int)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get_env_variable(self, name, data_type):
        value = self.env_vars.get(name)
        try:
            return data_type(value)
        except TypeError:
            print(f"Ошибка: Переменная среды {name} имеет неверный тип")

