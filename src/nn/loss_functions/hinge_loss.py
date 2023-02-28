import numpy as np
from nn.loss_functions.loss import Loss


def hinge_loss(inpt, target):
    """Реализует функцию ошибки hinge loss

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Список реальных классов
        Одномерный массив

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошибка
    """
    # Мы должны конвертировать массив реальных меток
    # в двумерный массив размера (N, C),
    # где N -- число элементов
    # С -- число классов
    C = inpt.array.shape[-1]
    target = np.eye(C)[target]

    # Можно взять такую реализацию - https://keras.io/api/losses/hinge_losses/#categoricalhinge-function
    pos = np.sum(target * inpt.array, axis=-1)
    neg = np.amax((1. - target) * inpt.array, axis=-1)
    loss = np.maximum(neg - pos + 1., 0.)
    loss = np.sum(loss) / inpt.array.shape[0]

    grad = (1. - target) * np.eye(C)[np.argmax((1. - target) * inpt.array, axis=-1)] - target
    grad = (grad.T * (neg - pos + 1. >= 0.)).T
    grad /= inpt.array.shape[0]

    return Loss(loss, grad, inpt.model)
