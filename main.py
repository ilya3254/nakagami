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
    plt.title('Функция распределения')
    plt.xlabel('Значение')
    plt.ylabel('Вероятность')
    plt.legend()
    plt.grid(True)
    plt.show()


"""МЕТОД МОМЕНТОВ"""
from scipy.special import gamma
from scipy.optimize import fsolve
# Заданные значения
X = -1.9115062633333346 + 2.518297  # == 0.6067907366666665
s = 0.05884687599947229

# Вспомогательная функция для решения системы уравнений
def equations_old(vars):
    mu, omega = vars
    eq1 = omega - (X**2 + s)
    eq2 = (gamma(mu + 0.5) / gamma(mu)) * np.sqrt((X**2 + s) / mu) - X
    return [eq1, eq2]

def equations(vars):
    mu, omega = vars
    eq1 = (gamma(mu + 0.5) / gamma(mu)) * np.sqrt(omega / mu) - X
    eq2 = omega * (1 - 1/mu * ((gamma(mu + 0.5) / gamma(mu))**2)) - s
    return [eq1, eq2]

# Начальные приближения
initial_guess = [0.5, 0.01]

# Решение системы уравнений
solution = fsolve(equations_old, initial_guess)

mu_solution, omega_solution = solution

print(f"ММ:\nmu = {mu_solution}")
print(f"omega = {omega_solution}\n")
#build_nakagami_cdf(nu=mu_solution, loc=omega_solution)


"""МЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ"""
from scipy.special import digamma

"""sum2 = 0
sumln = 0
for x in nakagami:
    #x += -1 * min(nakagami) + 0.1
    sum2 += x*x
    sumln += np.log(x)
print(sum2)
print(sumln)"""

for i in range(len(nakagami)):
    nakagami[i] *= -1
data = np.array(nakagami)

# Среднее квадратическое
mean_square = np.mean(data**2)

# Функция для поиска m
def equations(m):
    term1 = np.log(m)
    term2 = digamma(m)
    term3 = np.log(mean_square)
    term4 = np.mean(np.log(data))
    return term1 - term2 - term3 + term4

# Инициализация и решение
m_initial = 0.1  # Начальное приближение
mu_solution = fsolve(equations, m_initial)[0]

# Найденные параметры
omega_solution = mean_square
print(f"ММП:\nmu: {mu_solution}")
print(f"omega: {omega_solution}")
#build_nakagami_cdf(nu=mu_solution, loc=omega_solution)


"""ПОСТРОЕНИЕ ГРАФИКА ТЕОРЕТИЧЕСКИХ ФУНКЦИЯ С НАЙДЕННЫМИ ПАРАМЕТРАМИ
    И ЭМПИРИЧЕСКОЙ ФУНКЦИИ"""
def build_edf2(sample, size):
    """Построение эмпирической функции распределения"""
    values = np.random.choice(sample, size=size)
    data_sorted = np.sort(values)

    # Вычисляем значения ЭФР в точках, соответствующих отсортированным данным
    n = len(data_sorted)
    y_values = np.arange(1, n + 1) / n

    # Возвращаем данные для построения ЭФР
    return data_sorted, y_values


def build_nakagami_cdf2(nu, loc):
    """Построение функции распределения Нагами"""
    x = np.linspace(0, 3)  # 1000 точек для плавного графика
    y = nakagami_cdf(x, nu, loc)

    # Возвращаем данные для построения функции Нагами
    return x, y


def plot_combined(sample, size, params1, params2):
    """Объединенный график ЭФР и двух теоретических функций Нагами"""
    # Построение данных для ЭФР
    edf_x, edf_y = build_edf2(sample, size)

    # Построение данных для первой функции Нагами
    nakagami_x1, nakagami_y1 = build_nakagami_cdf2(*params1)

    # Построение данных для второй функции Нагами
    nakagami_x2, nakagami_y2 = build_nakagami_cdf2(*params2)

    # Визуализация обоих графиков на одном
    #plt.figure(figsize=(10, 6))

    # График ЭФР
    plt.step(edf_x, edf_y, where='post', label='ЭФР', color='blue')

    # График первой функции Нагами
    plt.plot(nakagami_x1, nakagami_y1, label=f'μ={params1[0]}, ω={params1[1]}', color='red')

    # График второй функции Нагами
    plt.plot(nakagami_x2, nakagami_y2, label=f'μ={params2[0]}, ω={params2[1]}', color='green')

    plt.title(f'Теоретические функции распределения и эмпирическая функция\n распределения для подвыборки из {size} элементов')
    plt.xlabel('x')
    plt.ylabel('Fn(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Пример использования функций
min123 = min(nakagami)
for i in range(len(nakagami)):
    nakagami[i] -= min123
size = 300  # Размер подвыборки
params1 = (1.6629, 0.427)  # Параметры для первой функции Нагами (ν, ω)
params2 = (0.87275, 3.7125)  # Параметры для второй функции Нагами (ν, ω)

#plot_combined(nakagami, size, params1, params2)


"""ГЕНЕРИРОВАНИЕ ВЫБОРКИ РАСПРЕДЕЛЕНИЯ ВЕЙБУЛЛА"""
import random
lambd = 1
k = 5
def num(u):
    return lambd*(-1*np.log(u))**(1/k)

veybull = []
for i in range(300):
    random_number = random.uniform(0, 1)
    veybull.append(num(random_number))

#build_edf(veybull, 300)
""""""

samples = np.array(nakagami)
"""8.2 + ММ и ММП"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nakagami, kstest, chi2_contingency

# Создание синтетической выборки
np.random.seed(42)
m_true = 2
Omega_true = 1
#samples = nakagami.rvs(m_true, scale=np.sqrt(Omega_true/m_true), size=1000)

# Проверка гипотезы о распределении с помощью критерия Колмогорова-Смирнова
def kolmogorov_smirnov_test(data, m, Omega):
    # Функция распределения Накагами
    cdf = lambda x: nakagami.cdf(x, m, scale=np.sqrt(Omega/m))
    statistic, p_value = kstest(data, cdf)
    return statistic, p_value

# Проверка гипотезы о распределении с помощью критерия χ^2
def chi_square_test(data, m, Omega, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_freq = nakagami.pdf(bin_centers, m, scale=np.sqrt(Omega/m)) * np.diff(bin_edges) * len(data)
    chi2_stat, p_value = chi2_contingency([hist, expected_freq])[:2]
    return chi2_stat, p_value

# Оценка параметров методом моментов
def estimate_parameters_moments(samples):
    sample_mean = np.mean(samples**2)
    sample_var = np.var(samples**2)
    Omega_mom = sample_mean
    m_mom = sample_mean**2 / sample_var
    return m_mom, Omega_mom

m_mom, Omega_mom = estimate_parameters_moments(samples)
print("\nММ: ", m_mom, Omega_mom)
#build_nakagami_cdf(nu=m_mom, loc=Omega_mom)

# Проверка гипотезы с помощью метода моментов
ks_stat_mom, ks_p_value_mom = kolmogorov_smirnov_test(samples, m_mom, Omega_mom)
chi2_stat_mom, chi2_p_value_mom = chi_square_test(samples, m_mom, Omega_mom)

# Оценка параметров методом максимального правдоподобия
from scipy.optimize import minimize

def mle_nakagami(params, data):
    m, Omega = params
    if m <= 0 or Omega <= 0:
        return np.inf  # возвращаем бесконечность, если параметры некорректны
    logpdf = nakagami.logpdf(data, m, scale=np.sqrt(Omega/m))
    if np.any(np.isnan(logpdf)):
        return np.inf  # возвращаем бесконечность, если есть некорректные значения
    return -np.sum(logpdf)

result = minimize(mle_nakagami, [1, 1], args=(samples,), bounds=[(0.5, None), (0.1, None)])
m_mle, Omega_mle = result.x

# Проверка гипотезы с помощью метода максимального правдоподобия
ks_stat_mle, ks_p_value_mle = kolmogorov_smirnov_test(samples, m_mle, Omega_mle)
chi2_stat_mle, chi2_p_value_mle = chi_square_test(samples, m_mle, Omega_mle)

# Вывод результатов
alpha = 0.05

print("Метод моментов:")
print(f"Колмогоров-Смирнов статистика: {ks_stat_mom}, p-значение: {ks_p_value_mom}")
print(f"χ^2 статистика: {chi2_stat_mom}, p-значение: {chi2_p_value_mom}")
print(f"Гипотеза отклоняется (К-С): {ks_p_value_mom < alpha}")
print(f"Гипотеза отклоняется (χ^2): {chi2_p_value_mom < alpha}")

print("\nМетод максимального правдоподобия:")
print(f"Колмогоров-Смирнов статистика: {ks_stat_mle}, p-значение: {ks_p_value_mle}")
print(f"χ^2 статистика: {chi2_stat_mle}, p-значение: {chi2_p_value_mle}")
print(f"Гипотеза отклоняется (К-С): {ks_p_value_mle < alpha}")
print(f"Гипотеза отклоняется (χ^2): {chi2_p_value_mle < alpha}")

# Построение графиков
x = np.linspace(0, 5, 300)
plt.figure(figsize=(10, 6))
plt.plot(x, nakagami.cdf(x, m_mom, scale=np.sqrt(Omega_mom/m_mom)), 'b-', label='Метод моментов')
plt.plot(x, nakagami.cdf(x, m_mle, scale=np.sqrt(Omega_mle/m_mle)), 'r--', label='Метод максимального правдоподобия')
plt.hist(samples, bins=30, density=True, cumulative=True, alpha=0.3, color='grey', label='Эмпирическая функция')
plt.title('Сравнение теоретических и эмпирической функций распределения')
plt.xlabel('Значения')
plt.ylabel('Функция распределения')
plt.legend()
#plt.show()


"""ПРИМЕР"""
# Хи-квадрат
# задаем количество интервалов
num_bins = 10

# разбиваем на интервалы и считаем частоту попадания значений в каждый интервал
freq, bins = np.histogram(np.sort(samples), bins=num_bins)

print('Частота попадания в интервалы: ', freq)
print('Границы интервалов: ', bins)

sum = 0
wait = 300/6
for i in range(num_bins):
    sum = sum + ((freq[i]-wait)**2)/wait
print("sum =",sum)
