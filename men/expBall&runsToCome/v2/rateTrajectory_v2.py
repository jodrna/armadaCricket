import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from paths import PROJECT_ROOT

# rate trajectory coefficients for 3_runsToComeSimClassOrd_v2.py, replacing the copied-in rateTrajectoryAdjustments.csv
# (an old output of OneDrive RAS_inputs/effect_of_runs_scored.py). The sim scales each ball by
#   runs:    1 + X_rr * (rateTrajectory - 1)        wickets: 1 + X_wr * (rateTrajectory - 1)
# where rateTrajectory = runs so far / par. The sim is ONE AVERAGE TEAM, so X must be pure in-innings momentum - how being
# ahead of expectation changes the next ball - with team, pitch and state quality removed:
#   - "ahead so far" and "next ball" are both measured against ovrexpr / ovrexpw, the per ball expected runs / wickets
#     given the state AND the players and ground, so known quality cancels out
#   - quality the ratings miss would still leak in (a team better than its ratings is ahead so far AND scores more next),
#     so the same slope is measured again with each innings' balls shuffled - shuffling keeps any innings level quality
#     but destroys the order, so ordered slope minus shuffled slope is the momentum effect. Both use identical definitions
#   - per ball slopes are made relative to the intercept, so the multiplier is exactly 1 at par (X = slope / intercept)
#   - slopes are smoothed across balls with a spline weighted by their standard errors
#   - a split half check on matches reports whether the effect is stable enough to trust

SEED = 42
N_SHUFFLES = 5                  # shuffles averaged, so the shuffled slope isn't itself noisy
MIN_BALLS_SO_FAR = 6            # no trajectory effect until there's at least an over of evidence (same as sim par being tiny early)
TRAJECTORY_CLIP = (0.25, 3.0)   # winsorise the so far ratio so a handful of extreme early starts don't set the slope

data = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv',
                   usecols=['matchID', 'ID', 'inningNumber', 'inningBallNumber', 'isValid', 'isWicket', 'batsmanRuns', 'ovrexpr', 'ovrexpw'])
data = data[(data['inningNumber'] == 1) & (data['isValid'] == 1)].dropna(subset=['ovrexpr', 'ovrexpw'])
data = data[(data['ovrexpr'] > 0) & (data['ovrexpw'] > 0)]
data = data.sort_values(['matchID', 'inningBallNumber', 'ID']).reset_index(drop=True)
data['batsmanRuns'] = data['batsmanRuns'].clip(lower=0)


def so_far_ratios(runs, expected, order):
    """Runs so far / expected runs so far before each ball, where 'so far' is the balls that come before it in `order`
    (a permutation of the innings' ball positions), always excluding the ball itself."""
    n = len(runs)
    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)
    cumRuns = np.concatenate([[0], np.cumsum(runs[order])])
    cumExpected = np.concatenate([[0], np.cumsum(expected[order])])
    # ball i (real position i) gets the first i balls of the ordering, swapping itself out for the next ball if it's among them
    ownInFirst = position < np.arange(n)
    idx = np.arange(n)
    soFarRuns = np.where(ownInFirst, cumRuns[idx + 1] - runs, cumRuns[idx])
    soFarExpected = np.where(ownInFirst, cumExpected[idx + 1] - expected, cumExpected[idx])
    with np.errstate(divide='ignore', invalid='ignore'):
        return soFarRuns / soFarExpected


def build_trajectories(frame, rng):
    ordered, shuffled = [], []
    for _, g in frame.groupby('matchID', sort=False):
        runs, wkts = g['batsmanRuns'].to_numpy(float), g['isWicket'].to_numpy(float)
        expR, expW = g['ovrexpr'].to_numpy(float), g['ovrexpw'].to_numpy(float)
        n = len(g)
        ordered.append(np.column_stack([so_far_ratios(runs, expR, np.arange(n)), so_far_ratios(wkts, expW, np.arange(n))]))
        sh = np.zeros((n, 2))
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(n)
            sh += np.column_stack([so_far_ratios(runs, expR, perm), so_far_ratios(wkts, expW, perm)])
        shuffled.append(sh / N_SHUFFLES)
    ordered, shuffled = np.vstack(ordered), np.vstack(shuffled)
    frame = frame.copy()
    # both slopes use the RUNS trajectory as the regressor - that's what the sim's rateTrajectory measures
    frame['rtOrdered'] = np.clip(ordered[:, 0], *TRAJECTORY_CLIP)
    frame['rtShuffled'] = np.clip(shuffled[:, 0], *TRAJECTORY_CLIP)
    frame['runsRatio'] = frame['batsmanRuns'] / frame['ovrexpr']
    frame['wicketRatio'] = frame['isWicket'] / frame['ovrexpw']
    return frame


def relative_slope(x, y):
    """OLS of y on (x - 1), returned as slope / intercept (so the multiplier is 1 at x = 1) with its standard error."""
    X = np.column_stack([np.ones(len(x)), x - 1])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (len(y) - 2)
    a, s = beta
    # delta method for s / a
    grad = np.array([-s / a ** 2, 1 / a])
    return s / a, np.sqrt(grad @ cov @ grad)


def momentum_by_ball(frame):
    rows = []
    # balls with nothing expected so far yet have no trajectory to measure
    frame = frame.dropna(subset=['rtOrdered', 'rtShuffled', 'runsRatio', 'wicketRatio'])
    for b, g in frame.groupby('inningBallNumber'):
        if b <= MIN_BALLS_SO_FAR or len(g) < 200:
            continue
        out = {'inningBallNumber': int(b), 'n': len(g)}
        for target, name in [('runsRatio', 'rr'), ('wicketRatio', 'wr')]:
            so, se_o = relative_slope(g['rtOrdered'].to_numpy(), g[target].to_numpy())
            ss, se_s = relative_slope(g['rtShuffled'].to_numpy(), g[target].to_numpy())
            out[f'X_{name}_ordered'], out[f'X_{name}_shuffled'] = so, ss
            out[f'X_{name}_raw'], out[f'X_{name}_se'] = so - ss, np.sqrt(se_o ** 2 + se_s ** 2)
        rows.append(out)
    return pd.DataFrame(rows)


def smooth(byBall, name):
    # weighted smoothing spline across balls (s = number of points is the standard choice with 1 / SE weights),
    # zero before there's enough evidence, flat beyond the last estimable ball
    spline = UnivariateSpline(byBall['inningBallNumber'], byBall[f'X_{name}_raw'], w=1 / byBall[f'X_{name}_se'], s=len(byBall))
    balls = np.arange(1, 121)
    values = spline(np.clip(balls, byBall['inningBallNumber'].min(), byBall['inningBallNumber'].max()))
    return np.where(balls <= MIN_BALLS_SO_FAR, 0.0, values)


rng = np.random.default_rng(SEED)
data = build_trajectories(data, rng)
byBall = momentum_by_ball(data)

output = pd.DataFrame({'inningBallNumber': np.arange(1, 121)})
output['X_rr_balls_smooth'] = smooth(byBall, 'rr')
output['X_wr_balls_smooth'] = smooth(byBall, 'wr')
output = output.merge(byBall, how='left', on='inningBallNumber')


# stability: refit on two random halves of matches - if the halves disagree the effect isn't real enough to use
matches = data['matchID'].unique()
halfA = set(np.random.default_rng(SEED + 1).permutation(matches)[:len(matches) // 2])
halves = {}
for label, mask in [('A', data['matchID'].isin(halfA)), ('B', ~data['matchID'].isin(halfA))]:
    hb = momentum_by_ball(data[mask])
    halves[label] = pd.DataFrame({'inningBallNumber': np.arange(1, 121), 'rr': smooth(hb, 'rr'), 'wr': smooth(hb, 'wr')})

print('\n=== rate trajectory coefficients (runs / wickets multiplier per unit of rateTrajectory - 1) ===')
show = output.set_index('inningBallNumber').loc[[12, 24, 36, 60, 90, 110, 120]]
print(show[['X_rr_ordered', 'X_rr_shuffled', 'X_rr_balls_smooth', 'X_wr_ordered', 'X_wr_shuffled', 'X_wr_balls_smooth']].round(3).to_string())
print('\n=== split half stability (smoothed, balls 7-120) ===')
for name in ['rr', 'wr']:
    a, b = halves['A'][name][MIN_BALLS_SO_FAR:], halves['B'][name][MIN_BALLS_SO_FAR:]
    print(f'X_{name}: half A mean {a.mean():+.3f}, half B mean {b.mean():+.3f}, correlation {np.corrcoef(a, b)[0, 1]:.2f}, '
          f'same sign on {np.mean(np.sign(a) == np.sign(b)):.0%} of balls')

output.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/rateTrajectoryAdjustments_v2.csv', index=False)
print('\nwritten auxiliaries/rateTrajectoryAdjustments_v2.csv')
