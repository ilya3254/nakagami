import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nakagami = pd.read_csv("../6. Распределение_Накагами_var_6.csv", header=None)
nakagami = nakagami.squeeze().tolist()


def summation(nakagami):
    """сумма элементов выборки"""
    summation = 0
    for value in nakagami:
        summation += value
    return summation


def average(nakagami):
    """выборочное среднее"""
    return summation(nakagami) / len(nakagami) if len(nakagami) > 0 else 0


def median(nakagami):
    """медиана"""
    n = len(nakagami)
    sorted_nakagami = sorted(nakagami)
    if n % 2 == 1:
        return sorted_nakagami[n // 2]
    else:
        return (sorted_nakagami[n // 2 - 1] + sorted_nakagami[n // 2]) / 2


def mode(nakagami):
    """мода"""
    frequency = {}
    for value in nakagami:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    max_frequency = max(frequency.values())
    mode = [key for key, val in frequency.items() if val == max_frequency]
    return mode if len(mode) == 1 else mode


def _range(nakagami):
    """размах выборки"""
    return max(nakagami) - min(nakagami)


def biased_variance(nakagami):
    """смещенная дисперсия"""
    avrg = average(nakagami)
    return sum((x - avrg) ** 2 for x in nakagami) / len(nakagami)


def unbiased_variance(nakagami):
    """несмещенная дисперсия"""
    avrg = average(nakagami)
    return sum((x - avrg) ** 2 for x in nakagami) / (len(nakagami) - 1)


def initial_moment(nakagami, k):
    """выборочный начальный момент k-ого порядка"""
    return sum(x ** k for x in nakagami) / len(nakagami)


def central_moment(nakagami, k):
    """выборочный центральный момент k-го порядка"""
    avrg = average(nakagami)
    return sum((x - avrg) ** k for x in nakagami) / len(nakagami)


def build_edf(sample, size):
    """построение эмпирической функции распределения"""
    values = np.random.choice(sample, size=size)
    data_sorted = np.sort(values)

    # Вычисляем значения ЭФР в точках, соответствующих отсортированным данным
    n = len(data_sorted)
    y_values = np.arange(1, n + 1) / n

    # Визуализация эмпирической функции распределения
    plt.step(data_sorted, y_values, where='post', label='ЭФР')
    plt.title(f'График эмпирической функции для подвыборки из {size} элементов')
    plt.xlabel('x')
    plt.ylabel('Fn(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


def build_hist(sample, size):
    """построение гистограммы для подвыборки из size элементов"""
    values = np.random.choice(sample, size=size)
    data_sorted = np.sort(values)
    plt.hist(data_sorted, bins='auto', edgecolor='black')
    plt.title(f'Гистограмма для подвыборки из {size} элементов')
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.show()


from scipy.special import gammainc
def nakagami_cdf(x, nu, loc):
    return gammainc(nu, (nu / loc) * x**2)

def build_nakagami_cdf(nu, loc):
    x = np.linspace(0,3)
    y = nakagami_cdf(x, nu, loc)
    #plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f'μ={nu}, ω={loc}')
    plt.title('Распределение Накагами')
    plt.xlabel('Значение')
    plt.ylabel('Вероятность')
    plt.legend()
    plt.grid(True)
    plt.show()

# Пример с случайными параметрами
build_nakagami_cdf(nu=2, loc=1)