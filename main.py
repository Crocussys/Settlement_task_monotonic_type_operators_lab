import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math


def operator_b(x):
    if x <= 1:
        return math.atan(x) + 3 * x
    elif 1 < x <= 4:
        return math.log(1 + x) + 3 + math.pi / 4 - math.log(2)
    else:
        return x ** 2 + 2 * x + math.pi / 4 - 21 - math.log(2 / 5)


def operator_a(x):
    return operator_b(x) + x


def f(x):
    return operator_a(x) + 2


def bisections(func, bounds, eps=1e-3):
    a, b = bounds
    if func(a) * func(b) > 0:
        print('На заданном интервале нет корней!')
        return
    x0 = (a + b) / 2
    error = abs(a - b)
    print(f"x: {x0} Точность: {error}")
    while error >= eps:
        if func(a) * func(x0) > 0:
            a = x0
        else:
            b = x0
        x0 = (a + b) / 2
        error = abs(a - b)
        print(f"x: {x0} Точность: {error}")
    return x0


def f2(x, y):
    return -(operator_a(y) + 2) / x


def runge_kutta(x0, y0, h=0.1, eps=1e-3):
    xs, ys = [x0], [y0]
    k1 = h * f2(x0, y0) / 3
    k2 = h * f2(x0 + h / 3, y0 + k1) / 3
    k3 = h * f2(x0 + h / 3, y0 + k1 / 2 + k2 / 2) / 3
    k4 = h * f2(x0 + h / 2, y0 + 3 / 8 * k1 + 9 / 8 * k3) / 3
    k5 = h * f2(x0 + h, y0 + 3 / 2 * k1 - 9 / 2 * k3 + 6 * k4) / 3
    y_next = y0 + (k1 + 4 * k4 + k5) / 2
    ys.append(y_next)
    x = x0 + h
    xs.append(x)
    i = 1
    print(f"{i}: {y_next}, {x}")
    y = y0
    while abs(y_next - y) >= eps:
        y = y_next
        k1 = h * f2(x, y) / 3
        k2 = h * f2(x + h / 3, y + k1) / 3
        k3 = h * f2(x + h / 3, y + k1 / 2 + k2 / 2) / 3
        k4 = h * f2(x + h / 2, y + 3 / 8 * k1 + 9 / 8 * k3) / 3
        k5 = h * f2(x + h, y + 3 / 2 * k1 - 9 / 2 * k3 + 6 * k4) / 3
        alpha = abs(k1 - 9 / 2 * k3 + 4 * k4 - k5 / 2)
        y_next = y + (k1 + 4 * k4 + k5) / 2
        ys.append(y_next)
        x += h
        xs.append(x)
        i += 1
        print(f"{i}: {y_next}, {x}")
        if alpha > 5 * eps:
            h /= 2
            continue
        if alpha < 5 / 32 * eps:
            h *= 2
    return xs, ys


def chart1(ax):
    x = [i / 100 for i in range(-5 * 100, 15 * 100 + 1)]
    x2 = [-2 for _ in range(len(x))]
    bx = list(map(operator_b, x))
    ax.plot(x, bx, label="Bx")
    ax.plot(x, x2, label="f")
    ax.plot(x, list(map(operator_a, x)), label="Ax")
    ax.legend()
    plt.xlim([-5, 10])
    plt.ylim([-5, 10])


def chart2(ax, xs, ys):
    ax.scatter(xs, ys)


def chart_main(chart_func, title, *args, **kwargs):
    fig, ax = plt.subplots()
    chart_func(ax, *args, **kwargs)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(which='major', color='gray')
    ax.grid(which='minor', color='gray', linestyle=':')
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.title(title)
    plt.savefig(f"{title}.svg".replace("\n", " "))
    plt.show()


if __name__ == '__main__':
    chart_main(chart1, "Заданные операторы")
    print(f"Ответ: {bisections(f, (-1, 0))}")
    xss, yss = runge_kutta(1, 4)
    chart_main(chart2, "Метод Рунге — Кутты\nНачальное приближение больше решения", xs=xss, ys=yss)
    print(f"Ответ: {yss[-1]}")
    xss, yss = runge_kutta(1, -4)
    chart_main(chart2, "Метод Рунге — Кутты\nНачальное приближение меньше решения", xs=xss, ys=yss)
    print(f"Ответ: {yss[-1]}")
