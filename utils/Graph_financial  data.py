import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import urllib.request


def bytespdate2num(b):
    return mdates.datestr2num(b.decode("utf-8"))


def graph_data(stock):

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    stock_price_url = (
        "https://www.quandl.com/api/v3/datasets/WIKI/"
        + stock
        + "/data.csv?api_key=TxcRm9NVPVc1STvdxtZQ"
    )

    source_code = urllib.request.urlopen(stock_price_url).read().decode("utf-8")

    stock_data = []
    split_source = source_code.split("\n")

    for line in split_source:
        split_line = line.split(",")
        if len(split_line) == 13:
            if "Date" not in line:
                stock_data.append(line)
    (
        date,
        openp,
        highp,
        lowp,
        closep,
        volume,
        blank,
        blank,
        blank,
        blank,
        blank,
        blank,
        blank,
    ) = np.loadtxt(
        stock_data, delimiter=",", unpack=True, converters={0: bytespdate2num},
    )

    ax1.plot_date(date, closep, "-", label="Price")
    # Put a rootation on xaxis labels
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    # Put a grid on it
    ax1.grid(True)

    # labeling

    plt.xlabel("date")
    plt.ylabel("price")
    plt.title("Important graph\nCheck it out!")

    # legends
    plt.legend()
    plt.subplots_adjust(
        left=0.09, bottom=0.15, right=0.94, top=0.90, wspace=0.2, hspace=0
    )

    plt.show()


graph_data("EBAY")
