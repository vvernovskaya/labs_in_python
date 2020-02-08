import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# y = a + b*x


def calc_b(x_data, y_data):
    sum_xy = 0
    for i in range(len(x_data)):
        sum_xy += x_data[i]*y_data[i]

    xy_avg = sum_xy / len(x_data)
    x_avg = sum(x_data) / len(x_data)
    y_avg = sum(y_data) / len(y_data)

    sum_x2 = 0
    for i in range(len(x_data)):
        sum_x2 += (x_data[i])**2

    x2_avg = sum_x2 / len(x_data)

    return (xy_avg - x_avg*y_avg) / (x2_avg - x_avg**2)


def calc_a(x_data, y_data):
    x_avg = sum(x_data) / len(x_data)
    y_avg = sum(y_data) / len(y_data)

    return y_avg - calc_b(x_data, y_data)*x_avg


def calc_b_error(x_data, y_data):
    x_avg = sum(x_data) / len(x_data)
    y_avg = sum(y_data) / len(y_data)

    sum_x2 = 0
    for i in range(len(x_data)):
        sum_x2 += (x_data[i]) ** 2

    x2_avg = sum_x2 / len(x_data)

    sum_y2 = 0
    for i in range(len(y_data)):
        sum_y2 += (y_data[i]) ** 2

    y2_avg = sum_y2 / len(y_data)

    return math.sqrt((1/len(x_data))*((y2_avg - y_avg**2) / (x2_avg -
                                                             x_avg**2) -
                                      calc_b(x_data, y_data)**2))


def calc_a_error(x_data, y_data):
    x_avg = sum(x_data) / len(x_data)

    sum_x2 = 0
    for i in range(len(x_data)):
        sum_x2 += (x_data[i]) ** 2

    x2_avg = sum_x2 / len(x_data)

    return calc_b_error(x_data, y_data) * (math.sqrt(x2_avg - x_avg**2))


def calc_avg_error(data):
    data_avg = sum(data) / len(data)

    # difference between each element from data and data average
    sum_each_avg_diff_2 = 0
    for i in range(len(data)):
        sum_each_avg_diff_2 += (data[i] - data_avg)**2

    n = len(data)
    return math.sqrt((1 / (n*(n-1)) * sum_each_avg_diff_2))


# table is the name of an excel table (example: 'table.xlsx'); sheet is the
# name of the sheet (example: 'Лист1'); column_num is the number of column
# to be extracted into an array: A = 0, B = 1 ...
def extract_data_into_array(table, sheet, column_num):
    pand_data = pd.read_excel(table, sheet)

    return pand_data.iloc[:, column_num].tolist()


# scatter plot with LSXY line
def make_diagram(x_data, y_data, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x=x_data, y=y_data, marker='o', c='r', edgecolor='b')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(xmin=min(x_data) - abs(max(x_data) - min(x_data))*0.1,
                xmax=max(x_data) + abs(max(x_data) - min(x_data))*0.1)
    ax.set_ylim(ymin=min(y_data) - abs(max(y_data) - min(y_data))*0.1,
                ymax=max(y_data) + abs(max(y_data) - min(y_data))*0.1)

    x = np.linspace(0, 10, 1000)
    ax.plot(x, calc_b(x_data, y_data) * x + calc_a(x_data, y_data),
            'r--')

    fig.tight_layout()

    return plt.show()

