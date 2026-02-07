"""
Burns-Mitchell Business Cycle Analysis for German GDP
======================================================
This script performs a Burns-Mitchell diagram analysis on German GDP data,
fits probability distributions to growth rates, and assesses statistical
significance through bootstrapping and student t-test.

For detailed methodology and theoretical background, see README.md

Author: Paul Gehlen
Data Source: FRED - CLVMNACSCAB1GQDE (German Real GDP, quarterly, seasonally adjusted)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pandas_datareader.data as web
from scipy.stats import t
import pickle

# Magic Numbers
WINDOW = 10
WINDOW_SIZE = 2 * WINDOW + 1
N_BOOTSTRAP = 5 ** 5
ALPHA = 0.05
N_BINS = 40
RANDOM_SEED = 42
BOOTSTRAP_BINS = 50
PDF_GRID_POINTS = 1000
OUTPUT_FILENAME1 = 'burns_mitchell.png'
OUTPUT_FILENAME2 = 'Lévy&Normal&Empirical_PDF.png'
OUTPUT_FILENAME3 = 'Bootstrap_Estimates_PDF'
OUTPUT_DPI = 300

def main():

    # Import Data and calculate growth rates
    peak = []
    data = web.DataReader("CLVMNACSCAB1GQDE", "fred", start="1991-01-01")
    growth_rate_data = (data['CLVMNACSCAB1GQDE']/data['CLVMNACSCAB1GQDE'].shift(1) - 1).to_frame('CLVMNACSCAB1GQDE')
    data_log = np.log(data)

    print(data_log.shape)

    # Localizing a peak using the metodology explained in README.md
    for i in range(WINDOW, len(data_log)-WINDOW):
        if (growth_rate_data.iloc[i-1, 0] > 0 and 
            growth_rate_data.iloc[i, 0] > 0 and 
            growth_rate_data.iloc[i+1, 0] < 0 and 
            growth_rate_data.iloc[i+2, 0] < 0):
            if (data_log.iloc[i-WINDOW, 0] is not None and not np.isnan(data_log.iloc[i-WINDOW, 0])) and (data_log.iloc[i+WINDOW, 0] is not None and not np.isnan(data_log.iloc[i+WINDOW, 0])):
                peak.append((i, data_log.iloc[i, 0]))
                continue

    userinput = 2 # User can choose which peak is getting viewed. Do not exceed the number of peaks in the list!

    print(f'Length of the Peak List={len(peak)}')

    # User Input validation
    if userinput < 0 or userinput >= len(peak):
        raise ValueError(f"Userinput must be between 0 and {len(peak)-1}")

    # Time Axis going from 10 quartes before to 10 quarters after the peak
    r = np.linspace(-WINDOW, WINDOW, num=WINDOW_SIZE, endpoint=True)

    # Finding the Data with regards to the peak finds and userinput
    index = peak[userinput][0]
    user_log_data = data_log.iloc[index-WINDOW: index+WINDOW+1, 0] * 100
    peak_range = []

    for i in range(0, len(peak)):
        start = peak[i][0] - WINDOW
        end = peak[i][0] + WINDOW + 1
        peak_window = data_log.iloc[start:end, 0].values
        peak_range.append((peak_window))

    peak_range_s = [[] for _ in range(WINDOW_SIZE)]

    for j in range(-WINDOW, WINDOW+1):  
        for i in range(len(peak)):
            peak_range_s[j+WINDOW].append(peak_range[i][j+WINDOW])

    # Calculating the Burns-Mitchell metrics as explained in the ReadMe
    y_s_sum = np.sum(np.array(peak_range_s), axis=1) 

    y_bar = np.sum(y_s_sum)/(len(peak)*WINDOW_SIZE)
    y_s_bar = y_s_sum / len(peak)
    averages = (y_s_bar - y_bar) * 100

    # Plotting the Burns-Mitchell Diagramm 
    fig_peak, axes = plt.subplots(1, 2, figsize=(14, 6))

    axe = axes[1]
    axe.plot(r, averages, '-r', alpha=0.7, linewidth=2, label=f'Trend of the Average')
    axe.scatter(0, averages[WINDOW], color='blue', s=100, zorder=5, label='Peak')

    axe.set_xlabel('Time')
    axe.set_ylabel('(Y_s_bar - Y_bar)*100')
    axe.set_title(f'Visualisation of the Average Peak Trend')
    axe.grid(alpha=0.3)
    axe.legend()

    ax = axes[0]
    ax.plot(r, user_log_data, '-r', alpha=0.7, linewidth=2, label=f'Visualisation of the Peak with index={index}')
    ax.scatter(0, peak[userinput][1]*100, color='blue', s=100, zorder=5, label='Peak')

    ax.set_xlabel('Time')
    ax.set_ylabel('100*log(Y)')
    ax.set_title(f'Visualisation of the peak with index={index}')
    ax.grid(alpha=0.3)
    ax.legend()

    fig_peak.tight_layout()
    fig_peak.savefig(OUTPUT_FILENAME1, dpi=OUTPUT_DPI)
    plt.show()

    # Comparison between Normal- and Lévy-fitted pdf on the growth rates
    growth_rate_data2 = pd.to_numeric(growth_rate_data.iloc[:, 0], errors='coerce')  
    growth_rate_data2 = growth_rate_data2.dropna()  
    growth_rate_data1 = np.sort(growth_rate_data2)

    # Fitting a Lévy Distribution with Parameters Alpha, Beta, Mü and Gamma on the Growth Rates
    params = stats.levy_stable.fit(growth_rate_data1)
    alpha, beta, loc, scale = params

    # Creating Grid for PDF visualisation
    i = np.linspace(start=min(growth_rate_data1), stop=max(growth_rate_data1),num=PDF_GRID_POINTS, endpoint=True)

    levy = stats.levy_stable.pdf(i, alpha, beta, loc, scale)

    # Lévy Parameter Output
    print(f'Alpha={alpha:.6f}, Beta={beta:.2f}, Mü={loc:.6f}, Gamma={scale:.6f}')

    # Fitting a Normal Distribution with Parameters Mü and Sigma squared on the Growth Rates
    norm = stats.norm.pdf(i, loc=growth_rate_data1.mean(), scale=growth_rate_data1.std())

    # Creating Plot for Normal- and Lévy-fitted Distribution of the Growth Rates
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]

    ax1.plot(i, norm, '-r', linewidth=2, label='Fitted Normal Distribution')
    ax1.hist(growth_rate_data1, bins=N_BINS, density=True, color='blue', alpha=0.3, label='Empirical Data')

    ax1.set_title('Fitted Normal Distribution vs. Data')
    ax1.set_xlabel('Growth rates')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    ax2 = axes[1]

    ax2.plot(i, levy, '-r', linewidth=2, label='Fitted Lévy-Distribution')
    ax2.hist(growth_rate_data1, bins=N_BINS, density=True, color='blue', alpha=0.3, label='Empirical Data')

    ax2.set_title('Fitted Lévy-Distribution vs. Data')
    ax2.set_xlabel('Growth rates')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME2, dpi=OUTPUT_DPI)
    plt.show()

    # Using Bootrstrapping to assess the statistical validity/signficance of the central Result in the Burns-Mitchell Diagramm
    peak_heights = []

    # Extract peak heights
    for i in range(len(peak_range)):
        window = peak_range[i]
        peak_heights.append(window[WINDOW])

    y_bar_0 = []

    # Calculate Deviation of every peak height from the Grand Mean
    for i in range(len(peak_heights)):
        peak_heights_i = peak_heights[i]
        y_bar_0.append((peak_heights_i-y_bar)*100)

    y_bar_0 = np.array(y_bar_0)

    # Point Estimation of Mü
    point_estimate = np.mean(y_bar_0)

    # Number of Bootstrap Simulations (5**5= Maximum number of unique Combinations with Order)
    n_bootstrap = N_BOOTSTRAP

    bootstrap_estimates = []

    np.random.seed(RANDOM_SEED) 
    for b in range(n_bootstrap):
        sample = np.random.choice(y_bar_0, size = len(y_bar_0), replace = True)
        bootstrap_estimates.append(np.mean(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    n = len(y_bar_0)

    # Calculating 95% Confidence Interval using the Bootstrap CI as well as t-test CI
    ci_1_b = np.percentile(bootstrap_estimates, 100*ALPHA/2)
    ci_2_b = np.percentile(bootstrap_estimates, 100*(1-ALPHA/2))
    ci_1_t = (point_estimate + t.ppf(ALPHA/2, df=n-1)*bootstrap_estimates.std()/np.sqrt(n))
    ci_2_t = (point_estimate - t.ppf(ALPHA/2, df=n-1)*bootstrap_estimates.std()/np.sqrt(n))

    # Output of the two CI's at Alpha=5% and 
    print(f'Point estimate={point_estimate:.6f} with Bootstrapping CI[{ci_1_b:.6f};{ci_2_b:.6f}]') 
    print(f'Point estimate={point_estimate:.6f} with t-Test CI[{ci_1_t:.6f};{ci_2_t:.6f}]') 

    # Visualisation of the Distribution of the Bootstrap Estimates
    plt.hist(bootstrap_estimates, bins=BOOTSTRAP_BINS, color='blue', label='Distribution of the Bootstrap Estimates', alpha=0.7, density = True)

    plt.xlabel('Values of the Bootstrap Estimates')
    plt.ylabel('Density')
    plt.title('Empirical Distribution of the Bootstrap Estimates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME3, dpi=OUTPUT_DPI)
    plt.show()

    # Save important Results for further Analysis
    ergebnisse = {
        'peak': peak, 
        'gdp': averages 
    }

    with open('peak_ergebnisse.pkl', 'wb') as f:
        pickle.dump(ergebnisse, f)

if __name__ == "__main__":
    main()