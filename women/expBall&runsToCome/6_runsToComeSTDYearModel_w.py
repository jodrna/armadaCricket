import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT
from sklearn.preprocessing import PolynomialFeatures


# import data and the runs to come modelled numbers
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/5_masterLookup_w.csv')





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







# fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True, sharey=True)
# axes = axes.flatten()  # flatten 5x2 into 1D array for easy looping
#
# for i in range(10):
#     ax = axes[i]
#     subset = masterLookup[masterLookup['totalInningWickets'] == i]
#
#     # Actual scatter (all daysGroup)
#     ax.scatter(
#         subset['totalInningRunsToComeSimBiasSpline'],
#         subset['totalInningRunsToComeSimSTD'],
#         s=10, alpha=0.6, label="Actual" if i == 0 else ""
#     )
#
#     # Predicted scatter (daysGroup == 10.5 only)
#     subset_pred = subset[subset['daysGroup'] == 16]
#     ax.scatter(
#         subset_pred['totalInningRunsToComeSimBiasSpline'],
#         subset_pred['totalInningRunsToComeSimSTDYear'],
#         s=10, alpha=0.6, color='orange', label="Predicted" if i == 0 else ""
#     )
#
#     ax.set_title(f"Wickets = {i}")
#
# # Global axis labels
# fig.text(0.5, 0.04, 'totalInningRunsToComeSimBiasSpline', ha='center')
# fig.text(0.04, 0.5, 'STD', va='center', rotation='vertical')
#
# # Add legend once
# axes[0].legend()
#
# plt.tight_layout(rect=[0.05, 0.05, 1, 1])
# plt.show()




masterLookup.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/6_masterLookup_w.csv', index=False)


ras_input = masterLookup.copy()
ras_input = ras_input[ras_input.daysGroup == 12.2]
ras_input['code'] = ras_input['totalInningWickets'] + ((121 - ras_input['inningBallNumber']) / 1000)

dfs = []

#process to smooth women's runs between wickets lost, as there aren't enough samples for the gaps between wickets lost to be sensible:

for x in range(1, 121):
    ras_input_copy = ras_input.copy()
    ras_input_copy = ras_input_copy[ras_input_copy.inningBallNumber == x]
    ras_input_copy = ras_input_copy.dropna(subset=['totalInningRunsToComeSimBiasSplineYearAdj', 'sample'])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(ras_input_copy[["totalInningWickets"]])
    model = LinearRegression().fit(X_poly, ras_input_copy["totalInningRunsToComeSimBiasSplineYearAdj"], sample_weight=ras_input_copy["sample"])
    ras_input_copy["totalInningRunsToComeSimBiasSplineYearAdj_smooth"] = model.predict(X_poly)
    dfs.append(ras_input_copy)

ras_input_new = pd.concat(dfs, ignore_index=True)

ras_input_new['totalInningRunsToComeSimBiasSplineYearAdj_smooth'] = np.where(ras_input_new['sample'] > 300, ras_input_new['totalInningRunsToComeSimBiasSplineYearAdj'], ((ras_input_new['totalInningRunsToComeSimBiasSplineYearAdj'] * ras_input_new['sample']) + (ras_input_new["totalInningRunsToComeSimBiasSplineYearAdj_smooth"] * (300 - ras_input_new['sample']))) / 300)

ras_input_new = ras_input_new.loc[:, ['code', 'sample', 'totalInningRunsToComeSimBiasSplineYearAdj_smooth', 'totalInningRunsToComeSimSTDYear', 'totalInningRunsToComeSimMin', 'totalInningRunsToComeSimMax', 'totalInningRunsToComeSimSkew', 'totalInningRunsToComeSimKurt']]

##have changed the output to non-adjusted runs for women, as I think the rate of run scoring isn't handled by the player ratings so the adjust reverts everything to average
## ^changed this back as the hypothesis about player ratings being wrong over time was wrong

ras_input_new.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/ras_input_innings_w.csv', index=False)