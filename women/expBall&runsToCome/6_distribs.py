import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from scipy.interpolate import UnivariateSpline
from scipy.stats import skew, kurtosis, pearson3, johnsonsu, norm, gaussian_kde
from paths import PROJECT_ROOT



# import data
trainData = pd.read_csv(PROJECT_ROOT / 'women/data/dataClean.csv', parse_dates=['date'])
# simsClass = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/ballSimsClass.csv')
simClassAdjusted = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/ballSimsClassRateAdjusted.csv')
# oldSTD = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/old_SD.csv')

# filter out the 2nd innings which we don't use in this instance
trainData = trainData[trainData['inningNumber'] == 1]



# # melt the old STD
# oldSTD = oldSTD.melt(id_vars='inningBallNumber',
#                    value_vars=[str(i) for i in range(6)],
#                    var_name='totalInningWickets',
#                    value_name='std')
# # Convert totalInningWickets to int (optional but likely preferred)
# oldSTD['totalInningWickets'] = oldSTD['totalInningWickets'].astype(int)



















# pivot stats
stdsReal = trainData.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'].agg(count='count', mean='mean', std='std', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)).reset_index()
# stdsClass = simsClass.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'].agg(count='count', mean='mean', std='std', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)).reset_index()
stdsClassAdjusted = simClassAdjusted.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'].agg(count='count', mean='mean', min='min', max='max', std='std', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)).reset_index()

# plot mean, std, skew for each wicket value and model
fig, axes = plt.subplots(10, 4, figsize=(18, 30), sharex=True)
metrics = ['mean', 'std', 'skew', 'kurtosis']
titles = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']

for i in range(10):  # totalInningWickets from 0 to 9
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        # Filter by count
        real_filtered = stdsReal[(stdsReal['totalInningWickets'] == i) & (stdsReal['count'] >= 100)]
        classAdjusted_filtered = stdsClassAdjusted[(stdsClassAdjusted['totalInningWickets'] == i) & (stdsClassAdjusted['count'] >= 1000)]

        # Plot Real and Adjusted
        ax.plot(real_filtered['inningBallNumber'], real_filtered[metric], label='Real', color='green')
        ax.plot(classAdjusted_filtered['inningBallNumber'], classAdjusted_filtered[metric], label='Class Adjusted', color='orange')

        # # Add oldSTD line only for std and for wickets 0–5
        # if metric == 'std' and i in oldSTD['totalInningWickets'].unique():
        #     old_filtered = oldSTD[oldSTD['totalInningWickets'] == i]
        #     ax.plot(old_filtered['inningBallNumber'], old_filtered['std'], label='Old STD', color='blue', linestyle='dashed')

        # Titles and labels
        if i == 0:
            ax.set_title(titles[j])
        if j == 0:
            ax.set_ylabel(f'Wickets: {i}')
        if i == 9 and j == 2:
            ax.legend(loc='lower right')

plt.tight_layout()
plt.show()






# # Plot mean, std, skew, kurtosis for each wicket value and model
# fig, axes = plt.subplots(10, 4, figsize=(18, 30), sharex=True)
# metrics = ['mean', 'std', 'skew', 'kurtosis']
# titles = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']
#
# for i in range(10):  # totalInningWickets from 0 to 9
#     for j, metric in enumerate(metrics):
#         ax = axes[i, j]
#
#         # Filter by wicket
#         real = stdsReal[stdsReal['totalInningWickets'] == i]
#         model = stdsClass[stdsClass['totalInningWickets'] == i]
#
#         # Use only inningBallNumbers where real has count >= 100
#         valid_balls = real[real['count'] >= 100]['inningBallNumber']
#         real_filtered = real[real['inningBallNumber'].isin(valid_balls)]
#         model_filtered = model[model['inningBallNumber'].isin(valid_balls)]
#
#         if not real_filtered.empty:
#             ax.plot(real_filtered['inningBallNumber'], real_filtered[metric], label='Real', color='green')
#             if not model_filtered.empty:
#                 ax.plot(model_filtered['inningBallNumber'], model_filtered[metric], label='Old', color='blue')
#
#         if i == 0: ax.set_title(titles[j])
#         if j == 0: ax.set_ylabel(f'Wickets: {i}')
#         if i == 9 and j == 2: ax.legend(loc='lower right')
#
# plt.tight_layout()
# plt.show()









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







# this will plot a real world distribution and then a distribution based off the 4 moments, mean, std, skew and kurt, define game state for the real world distribution
# Set game state
inningBallNumber = 114
totalInningWickets = 3

# Filter data and moments
trainData_filtered = trainData[(trainData['inningBallNumber'] == inningBallNumber) & (trainData['totalInningWickets'] == totalInningWickets)]
row = stdsReal.loc[(stdsReal['inningBallNumber'] == inningBallNumber) & (stdsReal['totalInningWickets'] == totalInningWickets)]
row2 = stdsClassAdjusted.loc[(stdsClassAdjusted['inningBallNumber'] == inningBallNumber) & (stdsClassAdjusted['totalInningWickets'] == totalInningWickets)]

# Extract the moments
mean = row['mean'].values[0]
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



