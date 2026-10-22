"""Feature transforms used by the selected production models."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline

RATING_COLUMNS = ['strikerRun','strikerWkt','nonStrikerRun','nonStrikerWkt','bowlerRun','bowlerWkt']
EVENTS = np.array([(r,w) for r in range(7) for w in range(2) if (r,w)!=(6,1)])

def features(d,ability=True,ball=False):
 x=pd.DataFrame(index=d.index)
 for c in ['runsRequired','inningBallsRemaining','totalInningWickets','ord','nonStrikerOrd']:x[c]=d[c].to_numpy(float)
 x['pressure']=x.runsRequired/x.inningBallsRemaining
 x['logPressure']=np.log(x.pressure.clip(lower=1e-6))
 x['logBalls']=np.log(x.inningBallsRemaining)
 x['ballsInOver']=(x.inningBallsRemaining-1)%6+1
 x['resourceZ']=(x.runsRequired-d['resourceMean'])/d['resourceStd']
 x['powerplay']=(x.inningBallsRemaining>=85).astype(int)
 x['death']=(x.inningBallsRemaining<=24).astype(int)
 x['orderGap']=x.totalInningWickets+2-x.ord
 x['tailOrder']=(x.ord-7).clip(lower=0)
 x['ballsWickets']=x.inningBallsRemaining*(10-x.totalInningWickets)
 if ball:
  x=x.drop(columns=['nonStrikerOrd','resourceZ'])
 if ability:
  rc=['strikerRun','strikerWkt','bowlerRun','bowlerWkt'] if ball else RATING_COLUMNS
  for c in rc:
   x[c]=np.log(d[c].clip(lower=1e-4))
   x[c+'Missing']=d[c].isna().astype(int)
  x['bowlertype_3']=d['bowlertype_3'].fillna('unknown').astype(str)
  if not ball:
   x['batRunDifference']=x.strikerRun-x.nonStrikerRun
   x['batWktDifference']=x.strikerWkt-x.nonStrikerWkt
  x['runMatchup']=x.strikerRun+x.bowlerRun
  x['wktMatchup']=x.strikerWkt+x.bowlerWkt
 return x

def spline_design(d,ability=False):
 runs=d.runsRequired.to_numpy(float);balls=d.inningBallsRemaining.to_numpy(float);w=d.totalInningWickets.to_numpy(float)
 z=(runs-d.resourceMean.to_numpy(float))/d.resourceStd.to_numpy(float)
 # State and fixed-resource coordinates with explicit missing-resource indicator.
 z=np.nan_to_num(z,nan=0,posinf=20,neginf=-10)
 vals=[np.log(runs),np.log(balls),w,d.ord.to_numpy(float),d.nonStrikerOrd.to_numpy(float),z]
 knots=[np.log([1,5,10,20,35,60,100,160,250,400]),np.log([1,3,6,12,24,48,72,84,96,120]),np.array([0,2,4,6,8,9]),np.array([1,3,5,7,9,11]),np.array([1,3,5,7,9,11]),np.array([-10,-4,-2,-1,0,1,2,4,10,20])]
 bases=[]
 for v,k in zip(vals,knots):
  t=np.r_[np.repeat(k[0],2),k,np.repeat(k[-1],2)]
  bases.append(sparse.csr_matrix(BSpline.design_matrix(np.clip(v,k[0],k[-1]),t,2)))
 blocks=[sparse.csr_matrix(np.ones((len(d),1)))]+bases
 for a,b in [(0,1),(0,2),(1,2),(0,3),(0,4),(1,3),(1,4),(2,3),(2,4),(5,1),(5,2)]:
  A=bases[a];B=bases[b]
  # Sparse row-wise tensor, preserving only the local active basis functions.
  rr=[];cc=[];vv=[]
  for i in range(A.shape[1]):
   block=B.multiply(A[:,i]).tocoo();rr.append(block.row);cc.append(block.col+i*B.shape[1]);vv.append(block.data)
  blocks.append(sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(d),A.shape[1]*B.shape[1])))
 extra=[d.resourceMean.isna().to_numpy(float)]
 if ability:
  for c in RATING_COLUMNS:
   rating=np.log(d[c].to_numpy(float));missing=~np.isfinite(rating);rating=np.nan_to_num(rating,nan=0)
   extra += [rating,missing.astype(float),rating*np.log(balls)/np.log(120),rating*np.clip(np.log(runs/balls),-2,2),rating*w/9]
 blocks.append(sparse.csr_matrix(np.column_stack(extra)))
 return sparse.hstack(blocks,format='csr')

