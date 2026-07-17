from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression
from paths import PROJECT_ROOT


data = pd.read_csv(
    PROJECT_ROOT / 'women/expBall&runsToCome/data/Cleaned_t20bbb3_adjusted_runs_to_come_w_2.csv',
    parse_dates=['date']
)

test = data.groupby(['ballsremaining'])['runs_to_come'].mean().reset_index()

data = data[data['innings'] == 1].copy()
data['runs_to_come'] = np.maximum(0, data['runs_to_come'] - data['RA_sum'])
data = data[data['ballsremaining'] > 2].copy()
data = data[data['year'] > 2020].copy()

# -------------------------
# Load hundred data
# -------------------------
hundredData = data[data.competition == 'The Hundred (Women\'s Comp)']
hundredData_test = hundredData[hundredData.ballsremaining == 3]
test = hundredData.groupby(['ballsremaining'])['runs_to_come'].mean().reset_index()
# -------------------------
# Load t20 data
# -------------------------
t20Data = data[data.competition != 'The Hundred (Women\'s Comp)']
t20Data_test = t20Data[t20Data.ballsremaining == 3]
test2 = t20Data.groupby(['ballsremaining'])['runs_to_come'].mean().reset_index()

# -------------------------
# Compare T20 vs Hundred by state
# -------------------------
hundredComparison = pd.pivot_table(
    hundredData,
    values='runs_to_come',
    index=['ballsremaining', 'wickets'],
    aggfunc=['mean', 'count']
).reset_index()

hundredComparison.columns = [
    'ballsremaining',
    'wickets',
    'mean_hundred',
    'count_hundred'
]

t20Comparison = pd.pivot_table(
    t20Data,
    values='runs_to_come',
    index=['ballsremaining', 'wickets'],
    aggfunc=['mean', 'count']
).reset_index()

t20Comparison.columns = [
    'ballsremaining',
    'wickets',
    'mean_t20',
    'count_t20'
]

comparison = hundredComparison.merge(
    t20Comparison,
    on=['ballsremaining', 'wickets'],
    how='left'
)

comparison = comparison[comparison['count_hundred'] > 5].copy()
comparison = comparison.dropna(subset=['mean_t20']).copy()

comparison['ratio_hundred_vs_t20'] = (
    comparison['mean_hundred']
    / comparison['mean_t20']
)

overall = comparison.copy()
overall['sum'] = overall['ratio_hundred_vs_t20'] * overall['count_hundred']
overall_ratio = overall['sum'].sum() / overall['count_hundred'].sum()

comparison['ratio_hundred_vs_t20'] = comparison['ratio_hundred_vs_t20'] / overall_ratio

X = comparison[['ballsremaining']]
y = comparison['ratio_hundred_vs_t20']
sample_weights = comparison['count_hundred']  # replace with your weights

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr', LinearRegression())
])

model.fit(X, y, lr__sample_weight=sample_weights)


comparison['ratio_pred'] = model.predict(
    comparison[['ballsremaining']]
)
# comparison['adj_runs_to_come'] = comparison['mean_hundred'] / comparison['ratio_pred']

comparison_result = comparison.drop_duplicates(subset=['ballsremaining'])

# # -------------------------
# # Merge comparison ratio to Hundred data for modelling
# # -------------------------
# hundredData = hundredData.merge(
#     comparison[['ballsremaining', 'wickets', 'mean_t20', 'mean_hundred', 'ratio_hundred_vs_t20']],
#     on=['ballsremaining', 'wickets'],
#     how='left'
# )
#
# hundredData = hundredData.dropna(
#     subset=['ratio_hundred_vs_t20']
# )
#
#
#
# # -------------------------
# # rates by ballsreamining only
# # -------------------------
# comparison_br = pd.pivot_table(
#     hundredData,
#     values=['mean_t20', 'mean_hundred', 'ratio_hundred_vs_t20'],
#     index='ballsremaining',
#     aggfunc='mean'
# ).reset_index()
#
#
# # -------------------------
# # Fit model on balls remaining
# # -------------------------
# X = comparison_br[['ballsremaining']]
# y = comparison_br['ratio_hundred_vs_t20']
#
# model = Pipeline([
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('lr', LinearRegression())
# ])
#
# model.fit(X, y)
#
# # -------------------------
# # Apply adjustment to Hundred data
# # -------------------------
# comparison_br['ratio_pred'] = model.predict(
#     comparison_br[['ballsremaining']]
# )
# comparison_br['adj_runs_to_come'] = comparison_br['mean_hundred'] / comparison_br['ratio_pred']
#
#
#
# # -------------------------
# # merge
# # -------------------------
# comparison = comparison.merge(
#     comparison_br,
#     on=['ballsremaining'],
#     how='left', suffixes=('', '_br')
# )
#
#
#
#
# # -------------------------
# # Plot T20, Hundred and adjusted curves
# # -------------------------
# plotData = comparison[
#     [
#         'ballsremaining',
#         'mean_t20_br',
#         'mean_hundred_br',
#         'adj_runs_to_come',
#         'ratio_hundred_vs_t20_br',
#         'ratio_pred'
#     ]
# ].drop_duplicates(
#     subset=['ballsremaining']
# ).sort_values(
#     'ballsremaining'
# )
#
# fig, ax1 = plt.subplots(figsize=(12, 7))
#
# # -------------------------
# # Runs-to-come curves
# # -------------------------
# ax1.plot(
#     plotData['ballsremaining'],
#     plotData['mean_t20_br'],
#     label='T20 runs to come',
#     linewidth=2
# )
#
# ax1.plot(
#     plotData['ballsremaining'],
#     plotData['mean_hundred_br'],
#     label='Hundred runs to come',
#     linewidth=2
# )
#
# ax1.plot(
#     plotData['ballsremaining'],
#     plotData['adj_runs_to_come'],
#     label='Adjusted Hundred runs to come',
#     linewidth=2
# )
#
# ax1.set_xlabel('Balls remaining')
# ax1.set_ylabel('Mean runs to come')
# ax1.grid(alpha=0.3)
#
# # -------------------------
# # Ratio axis
# # -------------------------
# ax2 = ax1.twinx()
#
# ax2.plot(
#     plotData['ballsremaining'],
#     plotData['ratio_hundred_vs_t20_br'],
#     '--',
#     linewidth=2,
#     label='Actual ratio'
# )
#
# ax2.plot(
#     plotData['ballsremaining'],
#     plotData['ratio_pred'],
#     ':',
#     linewidth=3,
#     label='Predicted ratio'
# )
#
# ax2.set_ylabel('Hundred / T20 ratio')
#
# # -------------------------
# # Combined legend
# # -------------------------
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
#
# ax1.legend(
#     lines1 + lines2,
#     labels1 + labels2,
#     loc='best'
# )
#
# plt.title('T20 vs Hundred Runs to Come')
# plt.tight_layout()
# plt.show()
#
#
#
# comparison.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/auxiliaries/hundredAdjusts.csv', index=False)
