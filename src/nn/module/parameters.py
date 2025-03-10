import numpy as np


class Parameters:
    """Здесь будут храниться параметры и их градиенты

    shape : tuple or int
        Определяет размер массива параметров
    """

    def __init__(self, shape):
        self.shape = shape
        self.params = np.zeros(shape)
        self.grads = np.zeros(shape)
        self.m = None
        self.v = None

    def _init_params(self, method='kaiming'):
        """Инициализация параметров

        ---------
        Параметры
        ---------
        method : str (default='kaiming')
            Метод инициализации параметров
            Пока доступен только 'kaiming'
        """
        if method == 'kaiming':
            self.params = np.random.normal(0, np.sqrt(2/self.shape[0]), self.shape)
