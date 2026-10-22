import numpy as np
import pandas as pd
from era_utils import decimal_year, atomic_csv, assign_splits
from paths import PROJECT_ROOT
from data_alignment_audit import audit_alignment


def main():
    d = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/dataClean.csv')
    d = d.loc[d.inningNumber.eq(2)].copy()
    audit = [{'check':'source_rows','value':len(d)}, {'check':'source_matches','value':d.matchID.nunique()}]
    conflict = d.groupby('matchID').chaseWin.nunique().gt(1)
    fractional = d.loc[d.runsRequired.notna() & ~np.isclose(d.runsRequired, d.runsRequired.round(), atol=1e-8, rtol=0), 'matchID']
    exclude = set(conflict.index[conflict]) | set(fractional)
    d = d.loc[~d.matchID.isin(exclude)].dropna(subset=['matchID','ID','runsRequired','inningBallsRemaining','totalInningWickets','ord','chaseWin'])
    d = d.loc[d.runsRequired.gt(0) & d.inningBallsRemaining.between(1,120) & d.totalInningWickets.between(0,9) & d.ord.between(1,11) & d.chaseWin.isin([0,1])].copy()
    assert not d.duplicated(['matchID','ID']).any()
    # Use an explicit provider-ID crosswalk. No performance or future match outcomes
    # are used in resolving identity. Names are only an unambiguous same-match fallback.
    p = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/auxiliaries/playerInfo.csv')
    p = p.dropna(subset=['cricinfo_id','playerid'])
    p = p.loc[~p.duplicated('cricinfo_id',keep=False)]
    d['nonPlayerID'] = d.nonstrikerid.map(p.set_index('cricinfo_id').playerid)
    names = d[['matchID','batsmanName','batterid']].drop_duplicates()
    names = names.loc[~names.duplicated(['matchID','batsmanName'],keep=False)]
    lookup = names.set_index(['matchID','batsmanName']).batterid
    fallback = pd.Series(lookup.reindex(pd.MultiIndex.from_arrays([d.matchID,d.nonstrikerName])).to_numpy(),index=d.index)
    d.nonPlayerID = d.nonPlayerID.fillna(fallback)
    orders = d.groupby(['matchID','batterid']).ord.min()
    d['nonStrikerOrd'] = orders.reindex(pd.MultiIndex.from_arrays([d.matchID,d.nonPlayerID])).to_numpy()
    newest = d.totalInningWickets+2
    d.nonStrikerOrd = d.nonStrikerOrd.fillna(pd.Series(np.where(d.ord.ne(newest),newest,np.nan),index=d.index))
    validpair = d.ord.le(newest) & d.nonStrikerOrd.between(1,11) & d.nonStrikerOrd.le(newest) & d.ord.ne(d.nonStrikerOrd)
    sorted_d = d.sort_values(['matchID','inningBallNumber','ID'])
    prevw = sorted_d.groupby('matchID').totalInningWickets.shift()
    preve = sorted_d.groupby('matchID').isWicket.shift()
    cum = sorted_d.groupby('matchID').isWicket.cumsum()-sorted_d.isWicket
    badrows = sorted_d.loc[sorted_d.totalInningWickets.lt(cum) | sorted_d.totalInningWickets.lt(prevw+preve),'ID']
    dup = sorted_d.duplicated(['matchID','inningBallNumber'],keep=False)
    badmatches = set(sorted_d.loc[dup,'matchID'])
    wc=sorted_d.groupby('matchID').isWicket.sum();badmatches |= set(wc.index[wc.gt(10)])
    d=d.loc[validpair & ~d.ID.isin(badrows) & ~d.matchID.isin(badmatches)].copy()
    for label,idcol in [('striker','batterid'),('nonStriker','nonPlayerID')]:
     r=pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/batRatings.csv')
     r=r.rename(columns={'matchid':'matchID','playerid':idcol,'run_rating_3':label+'Run','wkt_rating_3':label+'Wkt'})
     assert not r.duplicated(['matchID',idcol,'date']).any()
     d=d.merge(r,on=['matchID',idcol,'date'],how='left',validate='many_to_one')
    b=pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/deliveryBowlers.csv')
    b=b.rename(columns={'inningBallsRemaining':'bowlerSourceBalls'})
    d=d.merge(b[['matchID','ID','bowlerid','bowlerSourceBalls']],on=['matchID','ID'],how='left',validate='one_to_one')
    assert d.loc[d.bowlerid.notna(),'inningBallsRemaining'].eq(d.loc[d.bowlerid.notna(),'bowlerSourceBalls']).all()
    r=pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/bowlRatings.csv')
    r=r.rename(columns={'matchid':'matchID','playerid':'bowlerid','run_rating_3':'bowlerRun','wkt_rating_3':'bowlerWkt'})
    assert not r.duplicated(['matchID','bowlerid','date']).any()
    d=d.merge(r,on=['matchID','bowlerid','date'],how='left',validate='many_to_one')
    d['bowlertype_3']=d.bowlertype_3.fillna('unknown')
    for col in ['strikerRun','strikerWkt','nonStrikerRun','nonStrikerWkt','bowlerRun','bowlerWkt']:
     audit.append({'check':col+'_missing','value':int(d[col].isna().sum())})
     assert d.loc[d[col].notna(),col].gt(0).all()
    # Supplied resource means are fixed pre-existing lookup features, not refitted here.
    r=pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/auxiliaries/resourceExtras.csv')
    r=r.loc[r.inningNumber.eq(1)].copy();r['inningBallsRemaining']=121-r.inningBallNumber
    cols=['totalInningRunsToComeSimBiasSpline','totalInningRunsToComeSimSTD']
    r=r[['inningBallsRemaining','totalInningWickets']+cols].drop_duplicates()
    assert not r.duplicated(['inningBallsRemaining','totalInningWickets']).any()
    r=r.rename(columns={cols[0]:'resourceMean',cols[1]:'resourceStd'})
    d=d.merge(r,on=['inningBallsRemaining','totalInningWickets'],how='left',validate='many_to_one')
    audit.append({'check':'resource_lookup_missing_rows','value':int(d.resourceMean.isna().sum())})
    assert d.resourceStd.dropna().gt(0).all()
    r.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/models/resources.csv',index=False)
    alignment, examples = audit_alignment(d)
    alignment.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/outputs/dataAlignmentAudit.csv', index=False)
    excluded_alignment = alignment.loc[alignment.excludeAlignment]
    excluded_alignment.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/outputs/excludedAlignmentMatches.csv', index=False)
    examples.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/outputs/dataAlignmentExamples.csv', index=False)
    audit.append({'check':'excluded_alignment_matches','value':len(excluded_alignment)})
    audit.append({'check':'excluded_alignment_rows','value':int(excluded_alignment.rows.sum())})
    d = d.loc[~d.matchID.isin(excluded_alignment.matchID)].copy()
    matches = assign_splits(d)
    atomic_csv(matches, PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/matchSplits.csv')
    d = d.merge(matches[['matchID','matchRank','fold','split']],on='matchID',validate='many_to_one')
    d['overNumber'] = ((120-d.inningBallsRemaining)//6+1).astype(int)
    d['eraYear'] = decimal_year(d.date)
    d = d.sort_values(['matchID','inningBallNumber','ID']).reset_index(drop=True)
    atomic_csv(d, PROJECT_ROOT / 'men/matchMarket/hybridWin4/data/training.csv')
    audit.extend([{'check':'eligible_rows','value':len(d)}, {'check':'eligible_matches','value':d.matchID.nunique()}])
    atomic_csv(pd.DataFrame(audit), PROJECT_ROOT / 'men/matchMarket/hybridWin4/outputs/dataAudit.csv')
    print(pd.DataFrame(audit).to_string(index=False),flush=True)


if __name__=='__main__':main()
