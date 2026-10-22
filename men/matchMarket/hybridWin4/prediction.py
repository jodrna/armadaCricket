import numpy as np
import pandas as pd
from paths import PROJECT_ROOT
from direct_portable import load_direct, predict_direct
from ball_models import load_ball, predict_ball
from era_utils import decimal_year
from blend import load_production_bundle, apply_bundle, simulation_required
from simulation import simulate

ALIASES={'striker_run_rating_3':'strikerRun','striker_wkt_rating_3':'strikerWkt','nonstriker_run_rating_3':'nonStrikerRun','nonstriker_wkt_rating_3':'nonStrikerWkt','bowler_run_rating_3':'bowlerRun','bowler_wkt_rating_3':'bowlerWkt'}

def load_models():
 base = PROJECT_ROOT / 'men/matchMarket/hybridWin4'
 bundle = load_production_bundle()
 names = [name for name, weight in zip(bundle['directNames'], bundle['directWeights']) if weight > 1e-12]
 direct = {name: load_direct(base / 'models' / name) for name in names}
 ball_name = bundle['ballName']
 ball = load_ball(base / 'models' / ball_name)
 priors = pd.read_csv(base / f'models/{ball_name}_priors.csv')
 resource = pd.read_csv(base / 'models/resources.csv')
 extra = pd.read_csv(base / 'auxiliaries/resourceExtras.csv')
 configurations = [model['configuration'] for model in direct.values()] + [ball['configuration']]
 minimum = max(str(c['minimumDate']) for c in configurations)
 maximum = min(str(c['maximumDate']) for c in configurations)
 future_years = max(float(c.get('future_years', 0)) for c in configurations)
 cap = (pd.Timestamp(maximum) + pd.Timedelta(days=365.25*future_years)).strftime('%Y-%m-%d')
 return dict(bundle=bundle, direct=direct, ball=ball, priors=priors, resource=resource,
             extra=extra, minimumDate=minimum, maximumDate=maximum, eraCapDate=cap)

def predict(states,models=None,number_of_sims=1000,seed=42,include_simulation=False):
 if models is None:models=load_models()
 d=pd.DataFrame([states] if isinstance(states,dict) else states).rename(columns=ALIASES).copy()
 if d.columns.duplicated().any():raise ValueError('Duplicate input names; use either public rating names or internal names, not both')
 required=['runsRequired','inningBallsRemaining','totalInningWickets','ord','nonStrikerOrd']
 missing=[c for c in required if c not in d]
 if missing:raise ValueError('Missing inputs: '+', '.join(missing))
 if 'matchDate' not in d:raise ValueError('Missing input: matchDate (YYYY-MM-DD)')
 d=d.reset_index(drop=True)
 dates=pd.to_datetime(d.matchDate,errors='raise',utc=True)
 if dates.isna().any():raise ValueError('matchDate must not be missing')
 d['matchDate']=dates.dt.strftime('%Y-%m-%d')
 d['eraYear']=decimal_year(d.matchDate)
 for c in required:
  d[c]=pd.to_numeric(d[c],errors='raise')
  if not np.isfinite(d[c]).all():raise ValueError('Non-finite '+c)
  if not np.equal(d[c],np.floor(d[c])).all():raise ValueError('Expected integer '+c)
 if not d.inningBallsRemaining.between(0,120).all() or not d.totalInningWickets.between(0,10).all():raise ValueError('Invalid balls/wickets')
 for c in ALIASES.values():
  if c not in d:d[c]=np.nan
  d[c]=pd.to_numeric(d[c],errors='raise')
  if not d.loc[d[c].notna(),c].gt(0).all() or np.isinf(d[c]).any():raise ValueError(c+' must be positive or missing')
 if 'bowlertype_3' not in d:d['bowlertype_3']='unknown'
 d.bowlertype_3=d.bowlertype_3.fillna('unknown').astype(str)
 d=d[required+list(ALIASES.values())+['bowlertype_3','matchDate','eraYear']].copy()
 # Training has identified bowlers only at <=24 balls. Never extrapolate
 # bowling inputs into earlier horizons as if those effects were observed.
 d['bowlingInputsUsed']=d.inningBallsRemaining.le(24)&(d[['bowlerRun','bowlerWkt']].notna().any(axis=1)|d.bowlertype_3.ne('unknown'))
 d.loc[d.inningBallsRemaining.gt(24),['bowlerRun','bowlerWkt']]=np.nan
 d.loc[d.inningBallsRemaining.gt(24),'bowlertype_3']='unknown'
 live=d.runsRequired.gt(0)&d.inningBallsRemaining.gt(0)&d.totalInningWickets.lt(10)
 d.loc[~live,'bowlingInputsUsed']=False
 if not (d.loc[live,'ord'].between(1,11)&d.loc[live,'nonStrikerOrd'].between(1,11)&d.loc[live,'ord'].ne(d.loc[live,'nonStrikerOrd'])).all():raise ValueError('Current batters need valid distinct orders')
 if (d.loc[live,['ord','nonStrikerOrd']].max(axis=1)>d.loc[live,'totalInningWickets']+2).any():raise ValueError('Batting orders exceed wickets lost + 2')
 out=d[['matchDate']+required].copy();out['bowlingInputsUsed']=d.bowlingInputsUsed
 out['latestTrainingDate']=models['maximumDate']
 out['dateBeyondTraining']=d.matchDate.gt(models['maximumDate'])
 out['effectiveEraDate']=d.matchDate.clip(lower=models['minimumDate'],upper=models['eraCapDate'])
 out['eraDateClipped']=out.effectiveEraDate.ne(d.matchDate)
 for c in ['directProbability','rawSimulationProbability','calibratedSimulationProbability','directWeight','simulationWeight','matchWinProbability']:out[c]=np.nan
 terminal=d.loc[~live,'runsRequired'].le(0).astype(float)
 for c in ['directProbability','rawSimulationProbability','calibratedSimulationProbability','matchWinProbability']:
  out.loc[~live,c]=terminal
 out.loc[~live,'directWeight']=1.
 out.loc[~live,'simulationWeight']=0.
 if not live.any():return out
 f=d.loc[live].copy();indices=f.index
 f=f.merge(models['resource'],on=['inningBallsRemaining','totalInningWickets'],how='left',validate='many_to_one')
 components = pd.DataFrame(0., index=f.index, columns=models['bundle']['directNames'])
 for name, model in models['direct'].items():
  components[name] = predict_direct(model, f)
 raw_sim = None
 if simulation_required(models['bundle']) or include_simulation:
  if int(number_of_sims) != number_of_sims or number_of_sims < 1:
   raise ValueError('number_of_sims must be a positive integer')
  raw_sim = np.concatenate([simulate(models['ball'], f.iloc[start:start+32], models['priors'], models['extra'],
      n_sims=int(number_of_sims), seed=seed+start, predictor=predict_ball)
      for start in range(0, len(f), 32)])
 diagnostic = apply_bundle(models['bundle'], f, components, raw_sim)
 for column in diagnostic:
  out.loc[indices, column] = diagnostic[column].to_numpy()
 return out
