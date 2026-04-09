import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT


# import data and the runs to come modelled numbers
trainData = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/expBall&runsToCome/5_masterLookup.csv')





# 1) Train only on rows with complete data for features + target
train_mask = (
    masterLookup['totalInningRunsToComeSim'].notna() &
    masterLookup['totalInningWickets'].notna() &
    masterLookup['totalInningRunsToComeSimSTD'].notna()
)
X_train = masterLookup.loc[train_mask, ['totalInningRunsToComeSim', 'totalInningWickets']]
y_train = masterLookup.loc[train_mask, 'totalInningRunsToComeSimSTD']
model = LinearRegression().fit(X_train, y_train)

# 2) Predict where the prediction features are present
pred_mask = (
    masterLookup['totalInningRunsToComeSimBiasSplineYear'].notna() &
    masterLookup['totalInningWickets'].notna()
)
X_pred = masterLookup.loc[pred_mask, ['totalInningRunsToComeSimBiasSplineYear', 'totalInningWickets']] \
    .rename(columns={'totalInningRunsToComeSimBiasSplineYear': 'totalInningRunsToComeSim'})

masterLookup['totalInningRunsToComeSimSTDYear'] = np.nan
masterLookup.loc[pred_mask, 'totalInningRunsToComeSimSTDYear'] = model.predict(X_pred)

# 3) Compute group mean of predictions where available
pred_mean_by_state = (
    masterLookup
    .groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToComeSimSTDYear']
    .transform('mean')
)

# 4) Rescale predictions, preserving NaNs
scale = np.where(
    (pred_mean_by_state.notna()) & (pred_mean_by_state != 0),
    masterLookup['totalInningRunsToComeSimSTD'] / pred_mean_by_state,
    np.nan
)

masterLookup['totalInningRunsToComeSimSTDYear'] = masterLookup['totalInningRunsToComeSimSTDYear'] * scale







fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True, sharey=True)
axes = axes.flatten()  # flatten 5x2 into 1D array for easy looping

for i in range(10):
    ax = axes[i]
    subset = masterLookup[masterLookup['totalInningWickets'] == i]

    # Actual scatter (all daysGroup)
    ax.scatter(
        subset['totalInningRunsToComeSimBiasSpline'],
        subset['totalInningRunsToComeSimSTD'],
        s=10, alpha=0.6, label="Actual" if i == 0 else ""
    )

    # Predicted scatter (daysGroup == 10.5 only)
    subset_pred = subset[subset['daysGroup'] == 16]
    ax.scatter(
        subset_pred['totalInningRunsToComeSimBiasSpline'],
        subset_pred['totalInningRunsToComeSimSTDYear'],
        s=10, alpha=0.6, color='orange', label="Predicted" if i == 0 else ""
    )

    ax.set_title(f"Wickets = {i}")

# Global axis labels
fig.text(0.5, 0.04, 'totalInningRunsToComeSimBiasSpline', ha='center')
fig.text(0.04, 0.5, 'STD', va='center', rotation='vertical')

# Add legend once
axes[0].legend()

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()




# masterLookup.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/expBall&runsToCome/6_masterLookup.csv', index=False)


