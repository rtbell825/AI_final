# Ryan Bell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    '''
    data = data.drop('Date', axis=1)
    data = data.drop(0, axis=0)
    '''
    data.isnull().any()
    data = data.fillna(method='ffill')
    dataset = []
    for i in range(len(data['Date'])):
        dataset.append([i, data['Close'][i]])
    return dataset

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def print_stats(dataset):
    print(len(dataset))
    avg = 0
    for item in dataset:
        avg = avg + item[1]
    avg = avg / len(dataset)
    avg = round_half_up(avg, 2)
    print(avg)
    dev = 0
    for item in dataset:
        dev = dev + math.pow((item[1] - avg), 2)
    dev = dev * (1 / (len(dataset) - 1))
    dev = math.sqrt(dev)
    dev = round_half_up(dev, 2)
    print(dev)

def regression_n(beta_0, beta_1, dataset):
    sum = 0.0
    for item in dataset:
        sum = sum + math.pow((beta_0 + (beta_1 * item[0]) - item[1]), 2)
    sum = (1/len(dataset)) * sum
    return sum

def gradient_descent_sgd(beta_0, beta_1, dataset):
    rand = random.randint(0, len(dataset) - 1)
    #print(dataset[rand])
    x = 2 * (beta_0 + (beta_1 * dataset[rand][0]) - dataset[rand][1])
    y = 2 * (beta_0 + (beta_1 * dataset[rand][0]) - dataset[rand][1]) * dataset[rand][0]
    #print(x, y)
    return (x, y)

def sgd(dataset, T, eta):
    # print(dataset)
    x_bar = 0.0
    for item in dataset:
        x_bar = x_bar + item[0]
    x_bar = x_bar / len(dataset)
    stdx = 0.0
    for item in dataset:
        stdx = stdx + math.pow(item[0] - x_bar, 2)
    stdx = math.sqrt(stdx / (len(dataset) - 1))
    for item in dataset:
        item[0] = (item[0] - x_bar) / stdx

    # print("************", dataset)
    old_beta = [0, 0]
    # beta = (0, 0)
    t = 1
    while t <= T:
        gds = gradient_descent_sgd(old_beta[0], old_beta[1], dataset)
        beta_0 = old_beta[0] - (eta * gds[0])
        beta_1 = old_beta[1] - (eta * gds[1])
        # mse = regression(beta[0], beta[1])
        # print(t, round_half_up(old_beta[0], 2), round_half_up(old_beta[1], 2), round_half_up(mse, 2))
        #print(t, round_half_up(beta_0, 2), round_half_up(beta_1, 2),
        #      round_half_up(regression_n(beta_0, beta_1, dataset), 2))
        old_beta[0] = beta_0
        old_beta[1] = beta_1
        t = t + 1
    return old_beta

def predict(betas, year):
    #print(betas)
    return round_half_up(betas[0] + (betas[1]), 2)

def get_model(dataset):
    betas = sgd(copy.deepcopy(dataset), 50, 0.1)
    #betas = compute_betas(copy.deepcopy(dataset))
    return betas

def print_expectation(dataset, pred, label):
    for element in dataset:
        #print(element[0])
        plt.plot(element[0], element[1], marker='o', markersize=3, color='red')
    for element in pred:
        plt.plot(element[0], element[1], marker='o', markersize=3, color='blue')

    plt.title(label + ' 30 Day Closing Price Estimate')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.show()

def print_general(data):
    general = []
    for i in range(len(data[0])):
        sum = 0
        for j in range(len(data)):
            sum = sum + data[j][i]
        sum = sum / len(data)
        general.append(sum)
    for i in range(len(general)):
        #print(element[0])
        plt.plot(i, general[i], marker='o', markersize=3, color='black')
    plt.title('Predicted General Stock Market Growth/Decay')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.show()

def print_pregroup(points):
    names = ['APPL', 'F', 'GM', 'MANT', 'MRNA', 'RDS', 'WDC']
    colors = ['red', 'blue', 'blue', 'red', 'yellow', 'green', 'red']
    i = 0
    for point in points:
        plt.plot(point[0], point[1], marker='o', markersize=3, color=colors[i])
        plt.text(point[0], point[1], names[i])
        i = i + 1
    plt.title('Pre-group Company Plot')
    plt.xlabel('Known Rate')
    plt.ylabel('Predicted Rate')
    plt.show()

def get_averages(datasets):
    averages = []
    for dataset in datasets:
        average = []
        for i in range(len(dataset)):
            if i < 1:
                continue
            else:
                average.append(dataset[i][1] - dataset[i-1][1])
        averages.append(average)
    return averages

def compute_points(averages):
    points = []
    for dataset in averages:
        pre_sum = 0
        post_sum = 0
        for i in range(len(dataset)):
            if i < len(dataset) - 30:
                pre_sum = pre_sum + dataset[i]
            else:
                post_sum = post_sum + dataset[i]
        pre_sum = pre_sum / len(dataset) - 30
        post_sum = post_sum / 30
        points.append([pre_sum, post_sum])
    return points

def main():
    APPL_data = load_data('AAPL.csv')
    F_data = load_data('F.csv')
    GM_data = load_data('GM.csv')
    MANT_data = load_data('MANT.csv')
    MRNA_data = load_data('MRNA.csv')
    RDS_data = load_data('RDS-B.csv')
    WDC_data = load_data('WDC.csv')

    datasets = []
    datasets.append(APPL_data)
    datasets.append(F_data)
    datasets.append(GM_data)
    datasets.append(MANT_data)
    datasets.append(MRNA_data)
    datasets.append(RDS_data)
    datasets.append(WDC_data)

    #print(APPL_data)
    #print_stats(APPL_data)
    #print(predict(APPL_data, 20))
    #model = get_model(APPL_data)
    predictions = []
    for set in datasets:
        prediction = []
        for i in range(len(APPL_data) + 30):
            if i < 30:
                continue
            if i < len(APPL_data):
                model = get_model(set[i-30:i])
            else:
                #print(APPL_predictions)
                model = get_model(prediction[i-250:i - 30])
            prediction.append([i, predict(model, i)])
        predictions.append(prediction)
    #print_expectation(APPL_data, APPL_predictions, "APPL")
    averages = get_averages(datasets)
    print_general(averages)
    company_points = compute_points(averages)
    #print_pregroup(company_points)


main()