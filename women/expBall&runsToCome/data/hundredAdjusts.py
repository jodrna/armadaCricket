from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression
from paths import PROJECT_ROOT


# -------------------------
# Load  data
# -------------------------
data = pd.read_csv(
    PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv',
    parse_dates=['date']
)


# isolate the hundred data
hundredData = data[(data['competition'] == 'The Hundred (Women\'s Comp)')].copy()
hundredData = hundredData[hundredData['inningNumber'] == 1]
hundredData['totalInningRunsToCome'] = hundredData['totalInningRunsEnd'] - (hundredData['totalInningRuns'] - hundredData['totalRuns'])
hundredData = hundredData[hundredData['inningBallsRemaining'] > 0].copy()


# isolate t20 data
t20Data = data[(data['competition'] != 'The Hundred (Women\'s Comp)')].copy()
t20Data = t20Data[t20Data['inningNumber'] == 1]
t20Data = t20Data[t20Data['year'] > 2020].copy()
t20Data = t20Data[t20Data['inningBallsRemaining'] > 0].copy()


# -------------------------
# Compare T20 vs Hundred by state
# -------------------------
hundredComparison = pd.pivot_table(
    hundredData,
    values='totalInningRunsToCome',
    index=['inningBallsRemaining', 'totalInningWickets'],
    aggfunc=['mean', 'count']
).reset_index()

hundredComparison.columns = [
    'inningBallsRemaining',
    'totalInningWickets',
    'mean_hundred',
    'count_hundred'
]

t20Comparison = pd.pivot_table(
    t20Data,
    values='totalInningRunsToCome',
    index=['inningBallsRemaining', 'totalInningWickets'],
    aggfunc=['mean', 'count']
).reset_index()

t20Comparison.columns = [
    'inningBallsRemaining',
    'totalInningWickets',
    'mean_t20',
    'count_t20'
]

comparison = hundredComparison.merge(
    t20Comparison,
    on=['inningBallsRemaining', 'totalInningWickets'],
    how='left'
)

comparison = comparison[comparison['count_hundred'] > 5].copy()
comparison = comparison.dropna(subset=['mean_t20']).copy()

comparison['ratio_hundred_vs_t20'] = (
    comparison['mean_hundred']
    / comparison['mean_t20']
)





# -------------------------
# Merge comparison ratio to Hundred data for modelling
# -------------------------
hundredData = hundredData.merge(
    comparison[['inningBallsRemaining', 'totalInningWickets', 'mean_t20', 'mean_hundred', 'ratio_hundred_vs_t20']],
    on=['inningBallsRemaining', 'totalInningWickets'],
    how='left'
)

hundredData = hundredData.dropna(
    subset=['ratio_hundred_vs_t20']
)



# -------------------------
# rates by ballsreamining only
# -------------------------
comparison_br = pd.pivot_table(
    hundredData,
    values=['mean_t20', 'mean_hundred', 'ratio_hundred_vs_t20'],
    index='inningBallsRemaining',
    aggfunc='mean'
).reset_index()


# -------------------------
# Fit model on balls remaining
# -------------------------
X = comparison_br[['inningBallsRemaining']]
y = comparison_br['ratio_hundred_vs_t20']

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr', LinearRegression())
])

model.fit(X, y)

# -------------------------
# Apply adjustment to Hundred data
# -------------------------
comparison_br['ratio_pred'] = model.predict(
    comparison_br[['inningBallsRemaining']]
)
comparison_br['totalInningRunsToCome100Adj'] = comparison_br['mean_hundred'] / comparison_br['ratio_pred']



# -------------------------
# merge
# -------------------------
comparison = comparison.merge(
    comparison_br,
    on=['inningBallsRemaining'],
    how='left', suffixes=('', '_br')
)




# -------------------------
# Plot T20, Hundred and adjusted curves
# -------------------------
plotData = comparison[
    [
        'inningBallsRemaining',
        'mean_t20_br',
        'mean_hundred_br',
        'totalInningRunsToCome100Adj',
        'ratio_hundred_vs_t20_br',
        'ratio_pred'
    ]
].drop_duplicates(
    subset=['inningBallsRemaining']
).sort_values(
    'inningBallsRemaining'
)

fig, ax1 = plt.subplots(figsize=(12, 7))

# -------------------------
# Runs-to-come curves
# -------------------------
ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['mean_t20_br'],
    label='T20 runs to come',
    linewidth=2
)

ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['mean_hundred_br'],
    label='Hundred runs to come',
    linewidth=2
)

ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['totalInningRunsToCome100Adj'],
    label='Adjusted Hundred runs to come',
    linewidth=2
)

ax1.set_xlabel('Balls remaining')
ax1.set_ylabel('Mean runs to come')
ax1.grid(alpha=0.3)

# -------------------------
# Ratio axis
# -------------------------
ax2 = ax1.twinx()

ax2.plot(
    plotData['inningBallsRemaining'],
    plotData['ratio_hundred_vs_t20_br'],
    '--',
    linewidth=2,
    label='Actual ratio'
)

ax2.plot(
    plotData['inningBallsRemaining'],
    plotData['ratio_pred'],
    ':',
    linewidth=3,
    label='Predicted ratio'
)

ax2.set_ylabel('Hundred / T20 ratio')

# -------------------------
# Combined legend
# -------------------------
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='best'
)

plt.title('T20 vs Hundred Runs to Come')
plt.tight_layout()
plt.show()



# comparison.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/auxiliaries/hundredAdjusts.csv', index=False)
# comparison_br.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/auxiliaries/hundredAdjustsCompare.csv', index=False)


