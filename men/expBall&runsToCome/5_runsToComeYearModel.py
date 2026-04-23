import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT


# import data and the runs to come modelled numbers
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/4_masterLookup.csv')


trainData['totalInningRunsToCome'] = trainData['totalInningRunsToComeAdj']

# masterlookup 1 year only
masterLookupSingle = masterLookup.copy()
masterLookupSingle = masterLookupSingle.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)

# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData.drop_duplicates(subset=['matchID']), values=['totalInningRunsEnd'], index=['year'], aggfunc='mean').reset_index()

# runs to come by host for the adjustment
runsToComeHost = pd.pivot_table(trainData, values=['totalInningRunsToCome'], index=['host', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index()

# vs HOST Overall
trainData = trainData.merge(runsToComeHost, how='left', on=['host', 'totalInningWickets', 'inningBallNumber'])
trainData['vsHostOverall'] = trainData['totalInningRunsToCome_x'] / trainData['totalInningRunsToCome_y']
trainData = trainData.dropna(subset=['vsHostOverall'])

# Features and target variable
X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
y = trainData['vsHostOverall']

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
trainData['yearFactor'] = model.predict(X)


# merge the runs to come numbers overall then multiply by the year factor
trainData = trainData.merge(masterLookupSingle.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToComeSimBiasSpline']], how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData = trainData.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear'])

# check the training error
print(mean_absolute_error(trainData['totalInningRunsToCome_x'], trainData['totalInningRunsToComeSimBiasSplineYear']))



# predict for all years
masterLookup['daysGroup'] = masterLookup['year'] - 2015
# Create empty list to collect rows
rows = []

# Loop through each row in masterLookup
for _, row in masterLookup.iterrows():
    rows.append(row)
    if row['daysGroup'] == 10:
        dup = row.copy()
        dup['daysGroup'] = 10.5
        rows.append(dup)

# Rebuild the DataFrame
masterLookup = pd.DataFrame(rows)
X = masterLookup[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = model.predict(X)
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']



# export
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/5_masterLookup.csv', index=False)



#
# # Prepare the plot
# fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(40, 60), sharex=True, sharey=True)
# fig.tight_layout(pad=5.0)
#
# # Loop through each row (wickets lost) and each column (comparison metric)
# comparison_metrics = [
#     'totalInningRunsToComeSim',
#     'totalInningRunsToComeSimBias',
#     'totalInningRunsToComeSimBiasSpline',
#     'totalInningRunsToComeSimBiasSplineYear'
# ]
#
# for i, wickets in enumerate(range(10)):
#     filtered_df = masterLookup[masterLookup['totalInningWickets'] == wickets]
#     x = filtered_df['inningBallNumber']
#     y_actual = filtered_df['totalInningRunsToCome']
#
#     for j, metric in enumerate(comparison_metrics):
#         ax = axes[i, j]
#         y_metric = filtered_df[metric]
#
#         # Plot the actual and comparison metric lines
#         ax.plot(x, y_actual)
#         ax.plot(x, y_metric)
#
#         # Add titles and labels
#         if i == 0:
#             ax.set_title(metric, fontsize=10)
#         if j == 0:
#             ax.set_ylabel(f"Wickets {wickets}", fontsize=10)
#
#         # Add legend only for the first subplot in each row
#         if j == 0:
#             ax.legend(fontsize=8)
#
# # Add a global title and adjust layout
# # fig.suptitle("Comparison of Actual and Simulated Runs to Come", fontsize=16)
# plt.xlabel("Ball Number", fontsize=12)
# plt.ylabel("Runs to Come", fontsize=12)
#
# # Show plot
# plt.show()



