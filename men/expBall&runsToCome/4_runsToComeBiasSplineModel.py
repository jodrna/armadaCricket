import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, kurtosis, pearson3, johnsonsu, norm, gaussian_kde
from paths import PROJECT_ROOT


# import necessary data
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
simData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrd.csv')
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/2_masterLookup.csv')

# situationMoments = simData.groupby(['totalInningWickets', 'inningBallNumber'])['totalInningRunsToCome'] \
#     .agg(count='count', std='std', min='min', max='max', skew=lambda x: x.skew(), kurtosis=lambda x: kurtosis(x, fisher=True)) \
#     .rename(columns=lambda c: f"sim{c.capitalize()}" if c not in ['inningBallNumber', 'totalInningWickets'] else c) \
#     .reset_index()


totalInningValidBallsFaced = pd.pivot_table(simData, index=['simID'], values=['inningBallNumber'], aggfunc='max').reset_index()
totalInningValidBallsFaced = totalInningValidBallsFaced.rename(columns={'inningBallNumber': 'totalInningValidBallsFaced'})
simData = simData.merge(totalInningValidBallsFaced, how='left', on=['simID'])

simData['totalInningValidBallsFacedToCome'] = (simData['totalInningValidBallsFaced'] + 1) - simData['inningBallNumber']
simData['bowledOut'] = np.where(simData['totalInningValidBallsFaced'] < 120, 1, 0)


# inning aggregates
inningsReal = pd.pivot_table(trainData[trainData['inningNumber'] == 1], index=['date', 'home', 'away', 'competition'], values=['isWicket', 'totalRuns', 'batsmanRuns', 'noballRuns', 'wideRuns', 'byeRuns', 'isValid'], aggfunc='sum').reset_index()
# inningsSim = pd.pivot_table(simData, index=['simID'], values=['isWicket', 'isWicketInvalid', 'm_batsmanRunsBall', 'invalidRunsBall', 'byeRunsOver', 'isValid', 'totalRuns'], aggfunc='sum').reset_index()


# we need total inning runs to predict runs to come
# realSituationRunsToCome = pd.pivot_table(trainData, values=['totalInningRunsToCome', 'totalInningWicketsToCome', 'sample'], index=['inningBallNumber', 'totalInningWickets'],
#                                    aggfunc={'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean', 'sample': 'sum'}).reset_index()
# simSituationRunsToCome = pd.pivot_table(simData, values=['totalInningRunsToCome', 'sample', 'totalInningValidBallsFacedToCome', 'bowledOut'], index=['inningBallNumber', 'totalInningWickets'],
#                                   aggfunc={'totalInningRunsToCome': 'mean', 'sample': 'sum', 'totalInningValidBallsFacedToCome': 'mean', 'bowledOut': 'mean'}).reset_index()
# we need total inning runs to predict runs to come
realSituationRunsToCome = trainData.groupby(['inningBallNumber', 'totalInningWickets']).agg(
    totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
    totalInningWicketsToCome=('totalInningWicketsToCome', 'mean'),
    sample=('sample', 'sum')
).reset_index()

simSituationRunsToCome = simData.groupby(['inningBallNumber', 'totalInningWickets']).agg(
    totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
    sample=('sample', 'sum'),
    totalInningValidBallsFacedToCome=('totalInningValidBallsFacedToCome', 'mean'),
    bowledOut=('bowledOut', 'mean'),
    totalInningRunsToComeSimCount=('totalInningRunsToCome', 'count'),
    totalInningRunsToComeSimSTD=('totalInningRunsToCome', 'std'),
    totalInningRunsToComeSimSkew=('totalInningRunsToCome', lambda x: x.skew()),
    totalInningRunsToComeSimKurt=('totalInningRunsToCome', lambda x: kurtosis(x, fisher=True)),
    totalInningRunsToComeSimMin=('totalInningRunsToCome', 'min'),
    totalInningRunsToComeSimMax=('totalInningRunsToCome', 'max'),
).reset_index()




# merge real vs sim runs to come together
situationRunsToCome = pd.merge(simSituationRunsToCome, realSituationRunsToCome, left_on=['inningBallNumber', 'totalInningWickets'], right_on=['inningBallNumber', 'totalInningWickets'],
                         how='left', suffixes=('Sim', ''))
situationRunsToCome['simBias'] = situationRunsToCome['totalInningRunsToComeSim'] / situationRunsToCome['totalInningRunsToCome']



# declare an external runs to come to be appeneded too from the internal loop, we're going to adjust each wicket down so that there is no discrepency between sim and reality
# this is the first adjustment to the sim numbers
situationRunsToCome_adj1 = pd.DataFrame()
for x in range(10):
    i_situationRunsToCome = situationRunsToCome.copy()
    i_situationRunsToCome = i_situationRunsToCome[(situationRunsToCome['sample'] > 61) & (i_situationRunsToCome['totalInningWickets'] == x)]
    X = i_situationRunsToCome[['inningBallNumber']]
    y = i_situationRunsToCome['simBias']
    # Create and fit the rf regression model
    model = RandomForestRegressor(random_state=42, max_depth=3)
    model.fit(X, y.values)

    # Predict the ratio of real to sim using the model
    i_situationRunsToCome = situationRunsToCome.copy()    # reset to the original to get the less than 50 samples back
    i_situationRunsToCome = i_situationRunsToCome[(i_situationRunsToCome['totalInningWickets'] == x) &
                                                  ((situationRunsToCome['sample'] > 61) | (i_situationRunsToCome['totalInningWickets'] < 2))]
    X_pred = i_situationRunsToCome[['inningBallNumber']]
    y_pred = model.predict(X_pred)
    i_situationRunsToCome['m_simBias'] = y_pred

    # when before ball 110 apply the adjust, afterwards do not, it is not needed
    i_situationRunsToCome['totalInningRunsToComeSimBias'] = np.where(i_situationRunsToCome['inningBallNumber'] < 110,
                                                                 i_situationRunsToCome['totalInningRunsToComeSim'] / i_situationRunsToCome['m_simBias'],
                                                                 i_situationRunsToCome['totalInningRunsToComeSim'])
    situationRunsToCome_adj1 = pd.concat([situationRunsToCome_adj1, i_situationRunsToCome], axis=0)




# this is the 2nd and final adjustment which simply smoothes out some bumps in the previous
situationRunsToCome_adj2 = pd.DataFrame()
for i in range(10):
    # start
    i_situationRunsToCome_adj1 = situationRunsToCome_adj1.copy()
    i_situationRunsToCome_adj1 = i_situationRunsToCome_adj1[i_situationRunsToCome_adj1['totalInningWickets'] == i]

    # Example curve
    x = i_situationRunsToCome_adj1[['inningBallNumber']]
    y = i_situationRunsToCome_adj1[['totalInningRunsToComeSimBias']]

    if i == 0:
        smoothing_factor = 30  # Adjust for smoothness (larger is smoother)
    else:
        smoothing_factor = 10

    # Fit a spline to the inside
    spline = UnivariateSpline(x, y, s=smoothing_factor)

    # Generate the smoothed 'y' values
    smoothed_y = spline(x)

    # Add the smoothed values as a new column in your insideFrame
    i_situationRunsToCome_adj1['totalInningRunsToComeSimBiasSpline'] = smoothed_y
    situationRunsToCome_adj2 = pd.concat([situationRunsToCome_adj2, i_situationRunsToCome_adj1], axis=0)





# a final change, last balls are strange so just set them to actual
situationRunsToCome_adj2['totalInningRunsToComeSimBiasSpline'] = np.where(situationRunsToCome_adj2['inningBallNumber'] == 120,
                                                                          situationRunsToCome_adj2['totalInningRunsToCome'], situationRunsToCome_adj2['totalInningRunsToComeSimBiasSpline'])


# now make sure within each ball number there isn't a strange situation where 3 wickets is less than 6 wickets for example
situationRunsToCome_adj2 = situationRunsToCome_adj2.sort_values(by=['inningBallNumber', 'totalInningWickets']).reset_index(drop=True)
ref = situationRunsToCome_adj2.loc[:, ['totalInningWickets', 'inningBallNumber']]
situationRunsToCome_adj2 = situationRunsToCome_adj2.sort_values(by=['inningBallNumber', 'totalInningRunsToComeSimBiasSpline'], ascending=[True, False]).reset_index(drop=True)
situationRunsToCome_adj2['inningBallNumber'] = ref['inningBallNumber']
situationRunsToCome_adj2['totalInningWickets'] = ref['totalInningWickets']


# now we can merge the situation runs to come model numbers into the master lookup table which includes the ball by ball values
masterLookup = masterLookup.merge(situationRunsToCome_adj2.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToCome', 'totalInningRunsToComeSim', 'simBias',
                                                                   'm_simBias', 'totalInningRunsToComeSimBias', 'totalInningRunsToComeSimBiasSpline', 'totalInningRunsToComeSimSTD',
                                                                   'totalInningRunsToComeSimSkew', 'totalInningRunsToComeSimKurt', 'totalInningRunsToComeSimMin', 'totalInningRunsToComeSimMax',
                                                                   'totalInningValidBallsFacedToCome', 'bowledOut']],
                                  how='left', on=['totalInningWickets', 'inningBallNumber'])

# adjust 0 wickets lost start of the innings to actual for the first few balls
masterLookup['totalInningRunsToComeSimBiasSpline'] = np.where((masterLookup['inningBallNumber'] < 6) & (masterLookup['totalInningWickets'] == 0),
                                                              masterLookup['totalInningRunsToCome'], masterLookup['totalInningRunsToComeSimBiasSpline'])

# masterLookup = masterLookup.merge(situationMoments, how='left', on=['totalInningWickets', 'inningBallNumber'])

# analyse results
trainData = trainData.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToComeSim', 'totalInningRunsToComeSimBias',
                                                 'totalInningRunsToComeSimBiasSpline']].drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']),
                            how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData = trainData.dropna(axis=0, subset=['totalInningRunsToComeSim', 'totalInningRunsToComeSimBias',
                                                 'totalInningRunsToComeSimBiasSpline'])
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSim']))
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBias']))
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSpline']))


# export
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/4_masterLookup.csv', index=False)

