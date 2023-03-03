import time

import numpy as np

from nn.loss_functions.hinge_loss import hinge_loss
from optimization.adam_optimizer import Adam


def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониторинг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')


def gradient_check(x, y, neural_net, eps=10**(-5)):
    optimizer = Adam(neural_net.parameters())
    for param in neural_net.parameters():
        loss_function = hinge_loss(neural_net(x), y)
        optimizer.zero_grad()
        loss_function.backward()
        analytical_grad = param.grads.flatten()

        numerical_grad = []
        for flat_index in progress_bar(range(param.params.size)):
            index = np.unravel_index(flat_index, param.params.shape)
            param.params[index] += eps
            pos_loss = hinge_loss(neural_net(x), y).loss
            param.params[index] -= 2*eps
            neg_loss = hinge_loss(neural_net(x), y).loss
            param.params[index] += eps
            numerical_grad.append((pos_loss - neg_loss)/(2*eps))
        numerical_grad = np.array(numerical_grad)

        diff = np.linalg.norm(numerical_grad - analytical_grad)
        diff /= np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad) + eps
        if diff > eps:
            return False

    return True
