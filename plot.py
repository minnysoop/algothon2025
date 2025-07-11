import matplotlib.pyplot as plt  # import pyplot, not matplotlib directly
import pandas as pd

def load_prices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df.values.T

def graph(file_name):
    data = load_prices(file_name)

    x = range(data.shape[1])
    y = data[0]

    plt.plot(x, y)

    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Price history of first stock')

    plt.show()

graph('prices.txt')
