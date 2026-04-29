import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_absolute_error
from scipy.interpolate import UnivariateSpline
from scipy.stats import skew, kurtosis, pearson3, johnsonsu, norm, gaussian_kde
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import r2_score
from paths import PROJECT_ROOT



# import data
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
simClassAdjusted = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrd.csv')
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')
matchMarket = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookup.csv')
trainData = trainData[trainData['inningNumber'] == 1]




# --- global axis ranges ---
masterLookup = masterLookup[masterLookup['daysGroup'] == 11]
x_line_min = masterLookup['inningBallNumber'].min()
x_line_max = masterLookup['inningBallNumber'].max()
y_line_min = masterLookup[['totalInningRunsToComeSimBiasSpline',
                           'totalInningRunsToComeSimBiasSplineYear']].min().min()
y_line_max = masterLookup[['totalInningRunsToComeSimBiasSpline',
                           'totalInningRunsToComeSimBiasSplineYear']].max().max()

x_scatter_min = masterLookup['totalInningRunsToComeSimBiasSpline'].min()
x_scatter_max = masterLookup['totalInningRunsToComeSimBiasSpline'].max()
y_scatter_min = masterLookup['totalInningRunsToComeSimSTD'].min()
y_scatter_max = masterLookup['totalInningRunsToComeSimSTD'].max()

fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(16, 32), sharex=False, sharey=False)

for wk in range(10):
    ax_line = axes[wk, 0]
    ax_scatter = axes[wk, 1]

    # line plots
    d_line = masterLookup.loc[masterLookup['totalInningWickets'] == wk,
                              ['inningBallNumber',
                               'totalInningRunsToComeSimBiasSpline',
                               'totalInningRunsToComeSimBiasSplineYear']].sort_values('inningBallNumber')
    ax_line.plot(d_line['inningBallNumber'], d_line['totalInningRunsToComeSimBiasSpline'])
    ax_line.plot(d_line['inningBallNumber'], d_line['totalInningRunsToComeSimBiasSplineYear'])
    ax_line.set_xlim(x_line_min, x_line_max)
    ax_line.set_ylim(y_line_min, y_line_max)

    # scatter plots
    d_scatter = masterLookup.loc[masterLookup['totalInningWickets'] == wk,
                                 ['totalInningRunsToComeSimBiasSpline', 'totalInningRunsToComeSimSTD']]
    ax_scatter.scatter(d_scatter['totalInningRunsToComeSimBiasSpline'],
                       d_scatter['totalInningRunsToComeSimSTD'],
                       s=10, alpha=0.7)
    ax_scatter.set_xlim(x_scatter_min, x_scatter_max)
    ax_scatter.set_ylim(y_scatter_min, y_scatter_max)

plt.tight_layout()
plt.show()





# the below prints, mean, std, skew, kurt, std vs mean for real and sim, so 2 lines on each chart, 5 charts per row, 10 rows for wickets, x value is balls remaining, y is runs
# pivot stats
stdsReal = trainData.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'].agg(count='count', mean='mean', std='std', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)).reset_index()
stdsClassAdjusted = simClassAdjusted.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'].agg(count='count', mean='mean', std='std', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)).reset_index()

# plot mean, std, skew for each wicket value and model + 5th column scatter (std vs mean)
fig, axes = plt.subplots(10, 5, figsize=(22, 30), sharex=False)
metrics = ['mean', 'std', 'skew', 'kurtosis']
titles  = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis', 'STD vs Mean']

for i in range(10):  # totalInningWickets from 0 to 9
    # Filter by count (do once per row)
    real_filtered = stdsReal[(stdsReal['totalInningWickets'] == i) & (stdsReal['count'] >= 100)]
    classAdjusted_filtered = stdsClassAdjusted[(stdsClassAdjusted['totalInningWickets'] == i) & (stdsClassAdjusted['count'] >= 1000)]

    # First 4 columns: original line plots
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        # Plot Real and Adjusted
        ax.plot(real_filtered['inningBallNumber'], real_filtered[metric], label='Real', color='green')
        ax.plot(classAdjusted_filtered['inningBallNumber'], classAdjusted_filtered[metric], label='Class Adjusted', color='orange')

        # Titles and labels
        if i == 0:
            ax.set_title(titles[j])
        if j == 0:
            ax.set_ylabel(f'Wickets: {i}')
        if i == 9 and j == 2:
            ax.legend(loc='lower right')

    # 5th column: scatter of std vs mean
    ax_scatter = axes[i, 4]
    # Plot only if there is data after dropna
    rf = real_filtered[['mean', 'std']].dropna()
    cf = classAdjusted_filtered[['mean', 'std']].dropna()

    if not rf.empty:
        ax_scatter.scatter(rf['mean'], rf['std'], label='Real', alpha=0.6, color='green')
    if not cf.empty:
        ax_scatter.scatter(cf['mean'], cf['std'], label='Class Adjusted', alpha=0.6, color='orange')
    if rf.empty and cf.empty:
        ax_scatter.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_scatter.transAxes)

    if i == 0:
        ax_scatter.set_title(titles[4])
    # Optional: only one legend for the scatter, bottom row
    if i == 9:
        ax_scatter.legend(loc='lower right')

plt.tight_layout()
plt.show()











# # Set the game state filter
# inningBallNumber = 60
# totalInningWickets = 0
#
# # Filter the datasets
# trainData_filtered = trainData[
#     (trainData['inningBallNumber'] == inningBallNumber) &
#     (trainData['totalInningWickets'] == totalInningWickets)
# ]
# # simClassAdjusted_filtered = simClassAdjusted[
# #     (simClassAdjusted['inningBallNumber'] == inningBallNumber) &
# #     (simClassAdjusted['totalInningWickets'] == totalInningWickets)
# # ]
# # simsClass_filtered = simsClass[
# #     (simsClass['inningBallNumber'] == inningBallNumber) &
# #     (simsClass['totalInningWickets'] == totalInningWickets)
# # ]
#
# # Use your provided moments
# mean = 104.61
# std = 16.74235
# skew_val = 0.11402
# kurt_val = -0.04852  # excess kurtosis
#
# # Generate synthetic normal sample to fit a Johnson SU that approximates those moments
# np.random.seed(42)
# synthetic_data = np.random.normal(loc=mean, scale=std, size=100000)
#
# # Fit Johnson SU to synthetic data (approximating your moments)
# a, b, loc, scale = johnsonsu.fit(synthetic_data)
# johnson_dist = johnsonsu(a, b, loc=loc, scale=scale)
# johnson_samples = johnson_dist.rvs(size=100000)
#
# # Plot the distributions
# plt.figure(figsize=(10, 6))
#
# # KDEs
# sns.kdeplot(trainData_filtered['totalInningRunsToCome'], label='real', fill=True)
# # sns.kdeplot(simsClass_filtered['totalInningRunsToCome'], label='class', fill=True)
# sns.kdeplot(johnson_samples, label='johnsonsu approx', fill=True, linestyle='--')
#
# plt.xlabel('totalInningRunsToCome')
# plt.ylabel('Density')
# plt.title(f'Inning Ball {inningBallNumber}, Wickets {totalInningWickets}')
# plt.legend()
# plt.grid(True)
# plt.show()







# this will plot a real world distribution and then a distribution based off the 4 moments, mean, std, skew and kurt
# Set game state
inningBallNumber = 1
totalInningWickets = 0

# Filter data and moments
trainData_filtered = trainData[(trainData['inningBallNumber'] == inningBallNumber) & (trainData['totalInningWickets'] == totalInningWickets)]
row = stdsReal.loc[(stdsReal['inningBallNumber'] == inningBallNumber) & (stdsReal['totalInningWickets'] == totalInningWickets)]
row2 = stdsClassAdjusted.loc[(stdsClassAdjusted['inningBallNumber'] == inningBallNumber) & (stdsClassAdjusted['totalInningWickets'] == totalInningWickets)]

# Extract the moments
# mean = row2['mean'].values[0]
mean = 173.5

std = row2['std'].values[0]
skew_val = row2['skew'].values[0]
kurt_val = row2['kurtosis'].values[0]

# Build Gram–Charlier PDF
x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
z = (x - mean) / std
phi = norm.pdf(z)
H3 = z**3 - 3 * z
H4 = z**4 - 6 * z**2 + 3
gc_pdf = phi * (1 + (skew_val / 6) * H3 + (kurt_val / 24) * H4)
gc_pdf = np.maximum(gc_pdf, 0)
gc_pdf /= np.trapz(gc_pdf, x)

# Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(trainData_filtered['totalInningRunsToCome'], label='real', fill=True)
plt.plot(x, gc_pdf, label='gram-charlier approx (normalized)', linestyle='--')
plt.xlabel('totalInningRunsToCome')
plt.ylabel('Density')
plt.title(f'Inning Ball {inningBallNumber}, Wickets {totalInningWickets}')
plt.legend()
plt.grid(True)
plt.show()







# this finds the mean which gives the match market win % we want at the start of the 2nd innings, set the mean first, try different means to get the % you want
# YOU MUST CHANGE THE MEAN ABOVE FIRST
runs_range = np.arange(0, 301)

probabilities = []
for r in runs_range:

    lower = r - 0.5
    upper = r + 0.5

    mask = (x >= lower) & (x <= upper)

    if mask.sum() > 1:
        prob = np.trapz(gc_pdf[mask], x[mask])
    else:
        prob = 0

    probabilities.append(prob)

prob_df = pd.DataFrame({
    'runs': runs_range,
    'probability': probabilities
})

# Normalise in case of small numerical drift
prob_df['probability'] = prob_df['probability'] / prob_df['probability'].sum()


# Optional: cumulative probability
prob_df['cum_probability'] = prob_df['probability'].cumsum()

matchMarket = matchMarket[matchMarket['inningBallNumber'] == 1]
prob_df = prob_df.merge(matchMarket.loc[:, ['runsRequired', 'm_chaseWin%']], how='left', left_on='runs', right_on='runsRequired')
prob_df['probs'] = prob_df['m_chaseWin%'] * prob_df['probability']
print(np.sum(prob_df['probs']))



