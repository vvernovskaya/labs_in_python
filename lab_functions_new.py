import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import scipy.optimize as opt

'''
table is the name of an excel table (example: 'table.xlsx'); sheet is the
name of the sheet (example: 'Лист1'); column_num is the number of column
to be extracted into an array: A = 0, B = 1 ...
'''


def extract_data_into_array(table, sheet, column_num):
    pand_data = pd.read_excel(table, sheet)

    return pand_data.iloc[:, column_num].tolist()


def calc_avg_error(*columns):
    if len(columns) == 1:  # if the data for same conditions is in one column
        data = columns[0]
        data_avg = sum(data) / len(data)

        # difference between each element from data and data average
        sum_each_avg_diff_2 = 0
        for i in range(len(data)):
            sum_each_avg_diff_2 += (data[i] - data_avg) ** 2

        n = len(data)
        return math.sqrt((1 / (n * (n - 1)) * sum_each_avg_diff_2))

    else:
        n = 0  # if the data for the same conditions is in lines
        col_size = 0
        for column in columns:
            n += 1
            col_size = len(column)

        err_list = []
        avg_list = []

        for i in range(col_size):
            for column in columns:
                print(i)
                avg_list[i] += column[i]

            avg_list[i] /= n

            for column in columns:
                err_list[i] += (column[i] - avg_list[i]) ** 2

            err_list[i] = math.sqrt(1 / (n*(n-1)) * err_list[i])

        return err_list

# y = a + b * x -- linear model
# method argument can be either "lsxy" (least square minimization), or "chi_square" (chi-square minimization)


class Lab:
    def __init__(self, method, x_data, y_data, error_bar_x=None, error_bar_y=None, title=None,
                 x_label=None, y_label=None):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.x = None
        self.method = method
        self.x_data = x_data
        self.y_data = y_data
        self.error_bar_x = error_bar_x
        self.error_bar_y = error_bar_y
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.b = None
        self.a = None
        self.calc_b()
        self.calc_a()
        #self.b_error = None
        self.b_error = self.calc_b_error()
        #self.a_error = None
        self.a_error = self.calc_a_error()
        self.make_diagram()

    def chi_square(self, tup):
        y_data_div = np.array(self.y_data)
        y_data_err_div = np.array(self.error_bar_y)
        y_data_max = max(self.y_data)
        y_data_div = y_data_div/y_data_max
        y_data_err_div = y_data_err_div/y_data_max
        a, b = tup
        model = a + b * y_data_div
        chi_square = np.sum(((y_data_div - model) / y_data_err_div) ** 2)
        return chi_square

    def calc_b(self):
        if self.method == "chi_square":
            x0 = np.array([0, 0])
            result = opt.minimize(self.chi_square, x0)
            self.a, self.b = result.x * max(self.y_data)

        elif self.method == "lsxy":
            sum_xy = 0
            for i in range(len(self.x_data)):
                sum_xy += self.x_data[i] * self.y_data[i]

            xy_avg = sum_xy / len(self.x_data)
            x_avg = sum(self.x_data) / len(self.x_data)
            y_avg = sum(self.y_data) / len(self.y_data)

            sum_x2 = 0
            for i in range(len(self.x_data)):
                sum_x2 += (self.x_data[i]) ** 2

            x2_avg = sum_x2 / len(self.x_data)

            self.b = (xy_avg - x_avg * y_avg) / (x2_avg - x_avg ** 2)

        else:
            print("Method parameter should be either chi_square, or lsxy")  # MAKE AN EXCEPTION OUT OF THIS

    def calc_a(self):
        if self.method == "chi_square":
            return

        elif self.method == "lsxy":
            x_avg = sum(self.x_data) / len(self.x_data)
            y_avg = sum(self.y_data) / len(self.y_data)

            self.a = y_avg - self.b * x_avg

    def calc_b_error(self):
        x_avg = sum(self.x_data) / len(self.x_data)
        y_avg = sum(self.y_data) / len(self.y_data)

        sum_x2 = 0
        for i in range(len(self.x_data)):
            sum_x2 += (self.x_data[i]) ** 2

        x2_avg = sum_x2 / len(self.x_data)

        sum_y2 = 0
        for i in range(len(self.y_data)):
            sum_y2 += (self.y_data[i]) ** 2

        y2_avg = sum_y2 / len(self.y_data)

        return math.sqrt((1 / len(self.x_data)) * ((y2_avg - y_avg ** 2) / (x2_avg - x_avg ** 2) - self.b ** 2))

    def calc_a_error(self):
        x_avg = sum(self.x_data) / len(self.x_data)

        sum_x2 = 0
        for i in range(len(self.x_data)):
            sum_x2 += (self.x_data[i]) ** 2

        x2_avg = sum_x2 / len(self.x_data)

        return self.b_error * (math.sqrt(x2_avg - x_avg ** 2))

    def make_diagram(self):
        self.ax.scatter(x=self.x_data, y=self.y_data, marker='o', c='purple', edgecolor='b')
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_xlim(xmin=min(self.x_data) - abs(max(self.x_data) - min(self.x_data)) * 0.1,
                    xmax=max(self.x_data) + abs(max(self.x_data) - min(self.x_data)) * 0.1)
        self.ax.set_ylim(ymin=min(self.y_data) - abs(max(self.y_data) - min(self.y_data)) * 0.1,
                    ymax=max(self.y_data) + abs(max(self.y_data) - min(self.y_data)) * 0.1)

        #self.ax.set_xlim(0,
        #xmax=max(self.x_data) + abs(max(self.x_data) - min(self.x_data)) * 0.1)
        #self.ax.set_ylim(0,
        #ymax=max(self.y_data) + abs(max(self.y_data) - min(self.y_data)) * 0.1)

        self.x = np.linspace(min(self.x_data), max(self.x_data), 1000)
        self.ax.plot(self.x, self.b * self.x + self.a, 'r--')

        self.ax.errorbar(self.x_data, self.y_data, xerr=self.error_bar_x, yerr=self.error_bar_y, fmt='o', ecolor='black',
                    capsize=5, color='red', mec='b', mew=1)

        #plt.xticks(np.arange(0, 13, step=1))
        #plt.yticks(np.arange(0, 2.5, step=0.1))

        self.fig.tight_layout()


def show_all():
    plt.show()