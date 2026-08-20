from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT


# -------------------------
# Settings
# -------------------------
powerplay_split = 85
start_weight = 10


# -------------------------
# Load data
# -------------------------
data = pd.read_csv(
    PROJECT_ROOT / 'men/expBall&runsToCome/data/dataCleanNew.csv',
    parse_dates=['date']
)


# -------------------------
# Scale the Hundred balls
# -------------------------
data['inningBallsRemainingOriginal'] = data['inningBallsRemaining']

hundred_mask = data['competition'] == "The Hundred (Men's Comp)"

hundred_balls_remaining = data.loc[
    hundred_mask,
    'inningBallsRemaining'
]

data.loc[hundred_mask, 'inningBallsRemaining'] = np.round(
    np.where(
        hundred_balls_remaining >= 76,
        120 - ((100 - hundred_balls_remaining) * 35 / 24),
        84 - ((75 - hundred_balls_remaining) * 83 / 74)
    ),
    0
).astype(int)


# -------------------------
# Isolate the Hundred data
# -------------------------
hundredData = data[
    data['competition'] == "The Hundred (Men's Comp)"
].copy()

hundredData = hundredData[
    hundredData['inningNumber'] == 1
].copy()

hundredData = hundredData[
    hundredData['inningBallsRemaining'] > 0
].copy()


# -------------------------
# Isolate T20 data
# -------------------------
t20Data = data[
    data['competition'] != "The Hundred (Men's Comp)"
].copy()

t20Data = t20Data[
    t20Data['inningNumber'] == 1
].copy()

t20Data = t20Data[
    t20Data['year'] > 2020
].copy()

t20Data = t20Data[
    t20Data['inningBallsRemaining'] > 0
].copy()


# -------------------------
# Compare T20 vs Hundred by state
# -------------------------
hundredComparison = pd.pivot_table(
    hundredData,
    values='totalInningRunsToCome',
    index=[
        'inningBallsRemainingOriginal',
        'inningBallsRemaining',
        'totalInningWickets'
    ],
    aggfunc=['mean', 'count']
).reset_index()

hundredComparison.columns = [
    'inningBallsRemainingOriginal',
    'inningBallsRemaining',
    'totalInningWickets',
    'mean_hundred',
    'count_hundred'
]

t20Comparison = pd.pivot_table(
    t20Data,
    values='totalInningRunsToCome',
    index=[
        'inningBallsRemaining',
        'totalInningWickets'
    ],
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
    on=[
        'inningBallsRemaining',
        'totalInningWickets'
    ],
    how='left'
)

comparison = comparison[
    comparison['count_hundred'] > 5
].copy()

comparison = comparison.dropna(
    subset=['mean_t20']
).copy()

comparison['ratio_hundred_vs_t20'] = (
    comparison['mean_hundred']
    / comparison['mean_t20']
)


# -------------------------
# Merge state comparison back to Hundred data
# -------------------------
hundredData = hundredData.merge(
    comparison[
        [
            'inningBallsRemainingOriginal',
            'inningBallsRemaining',
            'totalInningWickets',
            'mean_t20',
            'mean_hundred',
            'ratio_hundred_vs_t20'
        ]
    ],
    on=[
        'inningBallsRemainingOriginal',
        'inningBallsRemaining',
        'totalInningWickets'
    ],
    how='left'
)

hundredData = hundredData.dropna(
    subset=['ratio_hundred_vs_t20']
).copy()


# -------------------------
# Hundred comparison by balls remaining
# -------------------------
hundredComparison_br = pd.pivot_table(
    hundredData,
    values=[
        'totalInningRunsToCome',
        'totalInningWickets'
    ],
    index=[
        'inningBallsRemainingOriginal',
        'inningBallsRemaining'
    ],
    aggfunc='mean'
).reset_index()

hundredComparison_br = hundredComparison_br.rename(
    columns={
        'totalInningRunsToCome': 'mean_hundred',
        'totalInningWickets': 'mean_wickets_hundred'
    }
)


# -------------------------
# T20 comparison by balls remaining
# -------------------------
t20Comparison_br = pd.pivot_table(
    t20Data,
    values=[
        'totalInningRunsToCome',
        'totalInningWickets'
    ],
    index='inningBallsRemaining',
    aggfunc='mean'
).reset_index()

t20Comparison_br = t20Comparison_br.rename(
    columns={
        'totalInningRunsToCome': 'mean_t20',
        'totalInningWickets': 'mean_wickets_t20'
    }
)


# -------------------------
# Merge balls-remaining comparisons
# -------------------------
comparison_br = hundredComparison_br.merge(
    t20Comparison_br,
    on='inningBallsRemaining',
    how='left'
)

comparison_br = comparison_br.dropna(
    subset=[
        'mean_t20',
        'mean_hundred'
    ]
).copy()

comparison_br['ratio_hundred_vs_t20'] = (
    comparison_br['mean_hundred']
    / comparison_br['mean_t20']
)

comparison_br = comparison_br.sort_values(
    'inningBallsRemaining'
).reset_index(
    drop=True
)


# -------------------------
# Powerplay model data
# 85 to 120 balls remaining
# -------------------------
powerplayData = comparison_br[
    comparison_br['inningBallsRemaining'] >= powerplay_split
].copy()

powerplayData = powerplayData.sort_values(
    'inningBallsRemaining'
).reset_index(
    drop=True
)

start_ball = powerplayData[
    'inningBallsRemaining'
].max()

start_ratio = powerplayData.loc[
    powerplayData['inningBallsRemaining'] == start_ball,
    'ratio_hundred_vs_t20'
].mean()

boundary_ratio = powerplayData.loc[
    powerplayData['inningBallsRemaining'] == powerplay_split,
    'ratio_hundred_vs_t20'
].mean()


# -------------------------
# Powerplay baseline
# Exact at the start and powerplay boundary
# -------------------------
powerplayData['powerplay_progress'] = (
    powerplayData['inningBallsRemaining']
    - powerplay_split
) / (
    start_ball
    - powerplay_split
)

powerplayData['powerplay_linear_base'] = (
    boundary_ratio
    + (
        start_ratio
        - boundary_ratio
    )
    * powerplayData['powerplay_progress']
)

powerplayData['powerplay_curve_feature'] = (
    powerplayData['powerplay_progress']
    * (
        1
        - powerplayData['powerplay_progress']
    )
)

powerplayData['powerplay_difference'] = (
    powerplayData['ratio_hundred_vs_t20']
    - powerplayData['powerplay_linear_base']
)


# -------------------------
# Weight the start of the innings
# -------------------------
powerplay_weights = np.ones(
    len(powerplayData)
)

powerplay_weights[
    powerplayData['inningBallsRemaining'] >= start_ball - 10
] = start_weight


# -------------------------
# Fit powerplay curve
# -------------------------
powerplay_curve_model = LinearRegression(
    fit_intercept=False
)

powerplay_curve_model.fit(
    powerplayData[
        ['powerplay_curve_feature']
    ],
    powerplayData[
        'powerplay_difference'
    ],
    sample_weight=powerplay_weights
)

powerplay_curve = powerplay_curve_model.coef_[0]

powerplay_delta = (
    start_ratio
    - boundary_ratio
)

powerplay_curve = np.clip(
    powerplay_curve,
    powerplay_delta,
    -powerplay_delta
)

powerplayData['ratio_pred_constrained'] = (
    powerplayData['powerplay_linear_base']
    + powerplay_curve
    * powerplayData['powerplay_curve_feature']
)


# -------------------------
# Post-powerplay model data
# 1 to 84 balls remaining
# -------------------------
postPowerplayData = comparison_br[
    comparison_br['inningBallsRemaining'] < powerplay_split
].copy()

postPowerplayData = postPowerplayData.sort_values(
    'inningBallsRemaining'
).reset_index(
    drop=True
)


# -------------------------
# Anchor ball 84 to the modelled value at ball 85
# -------------------------
boundary_prediction = powerplayData.loc[
    powerplayData['inningBallsRemaining'] == powerplay_split,
    'ratio_pred_constrained'
].iloc[0]

postPowerplayData['balls_from_boundary'] = (
    postPowerplayData['inningBallsRemaining']
    - (powerplay_split - 1)
)

postPowerplayData['ratio_difference_from_boundary'] = (
    postPowerplayData['ratio_hundred_vs_t20']
    - boundary_prediction
)


# -------------------------
# Fit linear post-powerplay slope
# Ball 84 is fixed to the prediction at ball 85
# -------------------------
post_powerplay_model = LinearRegression(
    fit_intercept=False
)

post_powerplay_model.fit(
    postPowerplayData[
        ['balls_from_boundary']
    ],
    postPowerplayData[
        'ratio_difference_from_boundary'
    ]
)

postPowerplayData['ratio_pred_constrained'] = (
    boundary_prediction
    + post_powerplay_model.predict(
        postPowerplayData[
            ['balls_from_boundary']
        ]
    )
)

postPowerplayData.loc[
    postPowerplayData['inningBallsRemaining'] == powerplay_split - 1,
    'ratio_pred_constrained'
] = boundary_prediction


# -------------------------
# Combine constrained models
# -------------------------
comparison_br['ratio_pred_constrained'] = np.nan

powerplay_predictions = powerplayData.set_index(
    'inningBallsRemaining'
)['ratio_pred_constrained']

post_powerplay_predictions = postPowerplayData.set_index(
    'inningBallsRemaining'
)['ratio_pred_constrained']

comparison_br.loc[
    comparison_br['inningBallsRemaining'] >= powerplay_split,
    'ratio_pred_constrained'
] = comparison_br.loc[
    comparison_br['inningBallsRemaining'] >= powerplay_split,
    'inningBallsRemaining'
].map(
    powerplay_predictions
)

comparison_br.loc[
    comparison_br['inningBallsRemaining'] < powerplay_split,
    'ratio_pred_constrained'
] = comparison_br.loc[
    comparison_br['inningBallsRemaining'] < powerplay_split,
    'inningBallsRemaining'
].map(
    post_powerplay_predictions
)


# -------------------------
# Apply constrained adjustment
# -------------------------
comparison_br['totalInningRunsToCome100Adj'] = (
    comparison_br['mean_hundred']
    / comparison_br['ratio_pred_constrained']
)


# -------------------------
# Merge constrained ratio into state comparison
# -------------------------
comparison = comparison.merge(
    comparison_br[
        [
            'inningBallsRemainingOriginal',
            'inningBallsRemaining',
            'ratio_pred_constrained'
        ]
    ],
    on=[
        'inningBallsRemainingOriginal',
        'inningBallsRemaining'
    ],
    how='left'
)


# -------------------------
# Overall plot data
# -------------------------
plotData = comparison_br[
    [
        'inningBallsRemaining',
        'inningBallsRemainingOriginal',
        'mean_t20',
        'mean_hundred',
        'totalInningRunsToCome100Adj',
        'ratio_hundred_vs_t20',
        'ratio_pred_constrained'
    ]
].sort_values(
    'inningBallsRemaining'
)


# -------------------------
# Chart 1
# Overall runs and constrained ratio
# -------------------------
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['mean_t20'],
    label='T20 runs to come',
    linewidth=2
)

ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['mean_hundred'],
    label='Hundred runs to come',
    linewidth=2
)

ax1.plot(
    plotData['inningBallsRemaining'],
    plotData['totalInningRunsToCome100Adj'],
    label='Adjusted Hundred runs to come',
    linewidth=2
)

ax1.set_xlabel('Scaled balls remaining')
ax1.set_ylabel('Mean runs to come')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    plotData['inningBallsRemaining'],
    plotData['ratio_hundred_vs_t20'],
    '--',
    linewidth=2,
    label='Actual ratio'
)

ax2.plot(
    plotData['inningBallsRemaining'],
    plotData['ratio_pred_constrained'],
    ':',
    linewidth=3,
    label='Constrained predicted ratio'
)

ax2.axvline(
    powerplay_split,
    linestyle=':',
    linewidth=2,
    alpha=0.5,
    label='Powerplay boundary'
)

ax2.set_ylabel('Hundred / T20 ratio')

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


# -------------------------
# Apply adjustment by wicket state
# -------------------------
comparison['mean_hundred_adj'] = (
    comparison['mean_hundred']
    / comparison['ratio_pred_constrained']
)


# -------------------------
# Chart 2
# Runs and constrained ratio by wicket state
# -------------------------
fig, axes = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(18, 24),
    sharex=True
)

axes = axes.flatten()

for wicket in range(10):
    ax1 = axes[wicket]

    wicketData = comparison[
        comparison['totalInningWickets'] == wicket
    ].copy()

    wicketData = wicketData.dropna(
        subset=[
            'mean_t20',
            'mean_hundred',
            'mean_hundred_adj',
            'ratio_hundred_vs_t20',
            'ratio_pred_constrained'
        ]
    )

    wicketData = wicketData.sort_values(
        'inningBallsRemaining'
    )

    ax1.plot(
        wicketData['inningBallsRemaining'],
        wicketData['mean_t20'],
        label='T20 runs to come',
        linewidth=2
    )

    ax1.plot(
        wicketData['inningBallsRemaining'],
        wicketData['mean_hundred'],
        label='Hundred runs to come',
        linewidth=2
    )

    ax1.plot(
        wicketData['inningBallsRemaining'],
        wicketData['mean_hundred_adj'],
        label='Adjusted Hundred runs to come',
        linewidth=2
    )

    ax1.set_xlabel('Scaled balls remaining')
    ax1.set_ylabel('Mean runs to come')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        wicketData['inningBallsRemaining'],
        wicketData['ratio_hundred_vs_t20'],
        '--',
        linewidth=2,
        label='Actual ratio'
    )

    ax2.plot(
        wicketData['inningBallsRemaining'],
        wicketData['ratio_pred_constrained'],
        ':',
        linewidth=3,
        label='Constrained predicted ratio'
    )

    ax2.axvline(
        powerplay_split,
        linestyle=':',
        linewidth=2,
        alpha=0.5,
        label='Powerplay boundary'
    )

    ax2.set_ylabel('Hundred / T20 ratio')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='best',
        fontsize=8
    )

    ax1.set_title(
        f'{wicket} wickets lost'
    )

fig.suptitle(
    'T20 vs Hundred Runs to Come by Wicket State',
    fontsize=18
)

plt.tight_layout()
plt.show()


# -------------------------
# Wickets plot data
# -------------------------
wicketsPlotData = comparison_br[
    [
        'inningBallsRemaining',
        'mean_wickets_t20',
        'mean_wickets_hundred'
    ]
].sort_values(
    'inningBallsRemaining'
)


# -------------------------
# Chart 3
# Average wickets lost
# -------------------------
plt.figure(figsize=(12, 7))

plt.plot(
    wicketsPlotData['inningBallsRemaining'],
    wicketsPlotData['mean_wickets_t20'],
    label='T20 average wickets lost',
    linewidth=2
)

plt.plot(
    wicketsPlotData['inningBallsRemaining'],
    wicketsPlotData['mean_wickets_hundred'],
    label='Hundred average wickets lost',
    linewidth=2
)

plt.xlabel('Scaled balls remaining')
plt.ylabel('Average wickets lost')
plt.title('T20 vs Hundred Average Wickets Lost')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# comparison.to_csv(
#     PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/hundredAdjusts.csv',
#     index=False
# )

# comparison_br.to_csv(
#     PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/hundredAdjustsCompare.csv',
#     index=False
# )


