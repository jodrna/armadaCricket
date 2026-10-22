"""Integer-event chase simulation with cached ball inference.

Only fitting-match population priors are used for unknown future players.
The supplied extras lookup contains means, not an empirical distribution:
invalid-delivery runs use adjacent-integer random rounding, with one automatic
penalty run and remaining runs treated as running; byes are a one-run event on
zero-batter-run, non-wicket legal deliveries. These are explicit approximations. Free hits,
which batter was run out, and future bowling allocations are not identified.

The last LEGAL delivery is integrated over its outcome distribution. Preceding
invalid attempts remain simulated: this is variance reduction, not a claim of
an exact deterministic final-over solution.
"""
import numpy as np
import pandas as pd
from features import EVENTS
from ball_models import predict_ball


def build_priors(train):
 rows=[]
 for order in range(1,12):
  s=train.loc[train.ord.eq(order),['matchID','batterid','strikerRun','strikerWkt']].drop_duplicates(['matchID','batterid']).dropna()
  rows.append({'kind':'bat','group':order,'run':s.strikerRun.median() if len(s) else 1.,'wkt':s.strikerWkt.median() if len(s) else 1.,'type':'unknown'})
 for over in range(1,21):
  s=train.loc[train.overNumber.eq(over),['matchID','bowlerid','bowlerRun','bowlerWkt','bowlertype_3']].drop_duplicates(['matchID','bowlerid']).dropna()
  # Bowling identities only available in the death overs. Outside observed
  # coverage use missing inputs, not falsely labelled average observed bowlers.
  rows.append({'kind':'bowl','group':over,'run':s.bowlerRun.median() if len(s) else np.nan,'wkt':s.bowlerWkt.median() if len(s) else np.nan,'type':s.bowlertype_3.mode().iloc[0] if len(s) else 'unknown'})
 return pd.DataFrame(rows)



def _extras_array(extras):
    table = np.zeros((121, 10, 4), dtype=float)
    x = extras.copy()
    if 'inningBallsRemaining' not in x:
        x['inningBallsRemaining'] = 121-x.inningBallNumber
    if 'inningNumber' in x:
        x = x.loc[x.inningNumber.eq(1)]
    x = x.drop_duplicates(['inningBallsRemaining', 'totalInningWickets'])
    for row in x.itertuples(index=False):
        b, w = int(row.inningBallsRemaining), int(row.totalInningWickets)
        if 1 <= b <= 120 and 0 <= w <= 9:
            table[b, w] = [row.byeRunsOver, row.isWicketInvalidOver,
                           row.isInvalidOver, row.invalidRunsOver]
    if not np.isfinite(table).all():
        raise ValueError('Extras lookup contains missing or non-finite values')
    if np.any((table[:, :, :3] < 0) | (table[:, :, :3] > 1)):
        raise ValueError('Extras probabilities must lie between zero and one')
    if np.any((table[:, :, 2] > 0) & (table[:, :, 3] < 1)):
        raise ValueError('An invalid delivery must award at least one run')
    return table


def simulate(model, states, priors, extras, n_sims=1000, seed=42,
             predictor=None, final_ball_expectation=True):
    """Return chase-win probabilities in input row order.

    predictor(model, frame) must return 13 joint (batter-runs, wicket)
    probabilities in EVENTS order. Set model configuration partner_context=True
    when the model uses non-striker features; its cache then preserves them.
    n_sims is the number of particles per state.
    """
    if predictor is None:
        predictor = predict_ball
    if int(n_sims) != n_sims or n_sims < 1:
        raise ValueError('n_sims must be a positive integer')
    n_sims = int(n_sims)
    states = states.reset_index(drop=True)
    n = len(states)
    if n == 0:
        return np.empty(0)
    initial_runs = states.runsRequired.to_numpy(float)
    if not np.allclose(initial_runs, np.round(initial_runs), atol=1e-8, rtol=0):
        raise ValueError('Discrete simulation requires integer runs required')
    initial_balls = states.inningBallsRemaining.to_numpy(int)
    initial_w = states.totalInningWickets.to_numpy(int)
    if ((initial_balls < 0) | (initial_balls > 120)).any():
        raise ValueError('Balls remaining must lie between 0 and 120')
    if ((initial_w < 0) | (initial_w > 10)).any():
        raise ValueError('Wickets lost must lie between 0 and 10')
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(n), n_sims)
    runs = np.repeat(np.round(initial_runs).astype(int), n_sims)
    balls = np.repeat(initial_balls, n_sims)
    wickets = np.repeat(initial_w, n_sims)
    striker = np.zeros(n*n_sims, dtype=np.int8)
    nonstriker = np.ones(n*n_sims, dtype=np.int8)
    payoff = np.zeros(n*n_sims, dtype=float)
    max_runs = max(1, int(np.max(initial_runs)))
    # Tokens 0/1 identify the two present batters; order+1 identifies a future
    # batter. Rating arrays make strike rotation exact without copying ratings.
    bat = priors.loc[priors.kind.eq('bat')].set_index('group')
    bat_run = np.ones((n, 13))
    bat_wkt = np.ones((n, 13))
    bat_order = np.tile(np.maximum(np.arange(13)-1, 1), (n, 1))
    bat_order[:, 0] = states.ord.to_numpy(int)
    bat_order[:, 1] = states.nonStrikerOrd.to_numpy(int)
    bat_run[:, :2] = states[['strikerRun', 'nonStrikerRun']].to_numpy(float)
    bat_wkt[:, :2] = states[['strikerWkt', 'nonStrikerWkt']].to_numpy(float)
    for order in range(2, 12):
        bat_run[:, order+1] = float(bat.loc[order, 'run'])
        bat_wkt[:, order+1] = float(bat.loc[order, 'wkt'])
    bowling = priors.loc[priors.kind.eq('bowl')].set_index('group')
    types = sorted(set(bowling['type'].fillna('unknown').astype(str)) |
                   set(states.bowlertype_3.fillna('unknown').astype(str)))
    type_code = {name: i for i, name in enumerate(types)}
    bow_run = np.zeros((n, 121))
    bow_wkt = np.zeros((n, 121))
    bow_type = np.zeros((n, 121), dtype=int)
    for b in range(1, 121):
        over = (120-b)//6+1
        bow_run[:, b] = bowling.loc[over, 'run']
        bow_wkt[:, b] = bowling.loc[over, 'wkt']
        bow_type[:, b] = type_code[str(bowling.loc[over, 'type'])]
    for i, b in enumerate(initial_balls):
        if b > 0:
            stop = ((b-1)//6)*6
            bow_run[i, stop+1:b+1] = states.bowlerRun.iloc[i]
            bow_wkt[i, stop+1:b+1] = states.bowlerWkt.iloc[i]
            bow_type[i, stop+1:b+1] = type_code[str(states.bowlertype_3.fillna('unknown').iloc[i])]
    year = states.eraYear.to_numpy(float)
    event_table = _extras_array(extras)
    # Integer cache keys replace sorting ten floating state columns on every
    # delivery. Each state fixes its date and current/future bowling profiles.
    cache = {}
    cached_probability = np.empty((32768, len(EVENTS)), dtype=float)
    cached_count = 0
    partner_context = str(model.get('configuration', {}).get('partner_context', False)).lower() in ['true', '1', '1.0']

    def probabilities(active):
        nonlocal cached_probability, cached_count
        key = (((idx[active]*121+balls[active])*(max_runs+1)+runs[active])*10+
               wickets[active])*13+striker[active]
        if partner_context:
            key = key*13+nonstriker[active]
        unique, inverse = np.unique(key, return_inverse=True)
        cached_index = np.fromiter((cache.get(int(k), -1) for k in unique), dtype=np.int64, count=len(unique))
        is_missing = cached_index < 0
        missing = unique[is_missing]
        if len(missing):
            t = missing.copy()
            if partner_context:
                ns = t % 13
                t //= 13
            s = t % 13
            t //= 13
            w = t % 10
            t //= 10
            r = t % (max_runs+1)
            t //= max_runs+1
            b = t % 121
            state = t//121
            if not partner_context:
                ns = np.ones(len(missing), dtype=int)
            frame = pd.DataFrame(dict(runsRequired=r, inningBallsRemaining=b,
                totalInningWickets=w, ord=bat_order[state, s], nonStrikerOrd=bat_order[state, ns],
                strikerRun=bat_run[state, s], strikerWkt=bat_wkt[state, s],
                nonStrikerRun=bat_run[state, ns], nonStrikerWkt=bat_wkt[state, ns], bowlerRun=bow_run[state, b],
                bowlerWkt=bow_wkt[state, b], resourceMean=0., resourceStd=1.,
                eraYear=year[state], bowlertype_3=[types[x] for x in bow_type[state, b]]))
            p = np.asarray(predictor(model, frame), dtype=float)
            if p.shape != (len(frame), len(EVENTS)) or not np.isfinite(p).all():
                raise ValueError('Invalid ball-model probability matrix')
            p = np.maximum(p, 0)
            p /= p.sum(axis=1, keepdims=True)
            end = cached_count+len(missing)
            if end > len(cached_probability):
                expanded = np.empty((max(end, len(cached_probability)*2), len(EVENTS)), dtype=float)
                expanded[:cached_count] = cached_probability[:cached_count]
                cached_probability = expanded
            cached_probability[cached_count:end] = p
            cached_index[is_missing] = np.arange(cached_count, end)
            cache.update((int(k), i) for i, k in enumerate(missing, cached_count))
            cached_count = end
        return cached_probability[cached_index[inverse]]

    for attempt in range(480):
        active = np.flatnonzero((runs > 0) & (balls > 0) & (wickets < 10))
        if not len(active):
            break
        e = event_table[balls[active], wickets[active]]
        is_invalid = rng.random(len(active)) < e[:, 2]
        invalid = active[is_invalid]
        if len(invalid):
            mean_runs = e[is_invalid, 3]
            gained = np.floor(mean_runs).astype(int)+(rng.random(len(invalid)) < (mean_runs % 1))
            runs[invalid] -= gained
            out = invalid[rng.random(len(invalid)) < e[is_invalid, 1]]
            wickets[out] += 1
            striker[out] = np.minimum(wickets[out]+3, 12)
            swap = invalid[(gained-1) % 2 == 1]
            striker[swap], nonstriker[swap] = nonstriker[swap].copy(), striker[swap].copy()
        legal = active[~is_invalid]
        if not len(legal):
            continue
        p = probabilities(legal)
        zero_mass = p[:, (EVENTS[:, 0] == 0) & (EVENTS[:, 1] == 0)].sum(axis=1)
        # Lookup byes are per attempt; put their expected total onto legal
        # deliveries, and award them only when the batter scores zero.
        ee = event_table[balls[legal], wickets[legal]]
        bye_conditional = np.minimum(ee[:, 0]/np.maximum((1-ee[:, 2])*zero_mass, 1e-12), 1.)
        final = balls[legal] == 1 if final_ball_expectation else np.zeros(len(legal), bool)
        if final.any():
            a = legal[final]
            target = runs[a]
            win = (p[final]*(EVENTS[:, 0][None, :] >= target[:, None])).sum(axis=1)
            win += (target == 1)*zero_mass[final]*bye_conditional[final]
            payoff[a] = win
            balls[a] = 0
        if (~final).any():
            a = legal[~final]
            pp = p[~final]
            event = (rng.random(len(a))[:, None] > pp.cumsum(axis=1)).sum(axis=1).clip(max=len(EVENTS)-1)
            gained = EVENTS[event, 0].copy()
            out = EVENTS[event, 1].astype(bool)
            byes = (gained == 0) & ~out & (rng.random(len(a)) < bye_conditional[~final])
            gained += byes
            runs[a] -= gained
            balls[a] -= 1
            dismissed = a[out]
            wickets[dismissed] += 1
            striker[dismissed] = np.minimum(wickets[dismissed]+3, 12)
            swap = a[(gained % 2 == 1) ^ (balls[a] % 6 == 0)]
            striker[swap], nonstriker[swap] = nonstriker[swap].copy(), striker[swap].copy()
    else:
        raise RuntimeError('Simulation exceeded 480 attempted deliveries')
    payoff[runs <= 0] = 1.
    return payoff.reshape(n, n_sims).mean(axis=1)
