"""Continuous bounded date transforms for the selected spline and ball model."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from features import spline_design as state_spline_design
from era_utils import recency_weights

def bounded_years(frame, configuration):
    return np.clip(frame.eraYear.to_numpy(float), float(configuration['minimumYear']),
                   float(configuration['maximumYear'])+float(configuration.get('future_years',0)))

def natural_basis(years, minimum, maximum):
    """Natural cubic basis: continuous values/slopes, linear boundary tails."""
    z=(np.asarray(years)-minimum)/max(maximum-minimum,1.)
    knots=np.linspace(0,1,5)
    def d(k):return (np.maximum(z-k,0)**3-np.maximum(z-knots[-1],0)**3)/(knots[-1]-k)
    last=d(knots[-2])
    return np.column_stack([z]+[d(k)-last for k in knots[:-2]])

def normalized_basis(raw, configuration, fitting=False, weights=None):
    result=raw.copy()
    for j in range(raw.shape[1]):
        if fitting:
            mean=np.average(raw[:,j],weights=weights)
            scale=max(np.sqrt(np.average((raw[:,j]-mean)**2,weights=weights)),.03)
            configuration[f'curveMean_{j}']=float(mean)
            configuration[f'curveScale_{j}']=float(scale)
        result[:,j]=(raw[:,j]-float(configuration[f'curveMean_{j}']))/float(configuration[f'curveScale_{j}'])
    return result

def direct_design(frame,configuration,fitting=False):
    base=state_spline_design(frame,ability=True)
    years=bounded_years(frame,configuration)
    raw=natural_basis(years,float(configuration['minimumYear']),float(configuration['maximumYear']))
    normalized=normalized_basis(raw,configuration,fitting,recency_weights(frame,float(configuration['half_life'])))
    basis=sparse.csr_matrix(normalized)
    balls=frame.inningBallsRemaining.to_numpy(float)
    interactions=[np.ones(len(frame)),np.log(balls)/np.log(120),
                  np.clip(np.log(frame.runsRequired.to_numpy(float)/balls),-2,2),frame.totalInningWickets.to_numpy(float)/9]
    return sparse.hstack([base]+[basis.multiply(v[:,None]) for v in interactions],format='csr'),base.shape[1]

def curve_design(frame, configuration, fitting=False, weights=None):
    years = bounded_years(frame, configuration)
    raw = natural_basis(years, float(configuration['minimumYear']), float(configuration['maximumYear']))
    raw = normalized_basis(raw, configuration, fitting, weights)
    return np.column_stack([np.ones(len(frame)), raw])

def fit_model(train, configuration):
    configuration = dict(configuration)
    configuration.update(minimumYear=float(train.eraYear.min()), maximumYear=float(train.eraYear.max()),
                         minimumDate=str(train.date.min()), maximumDate=str(train.date.max()),
                         trainingRows=len(train), trainingMatches=train.matchID.nunique())
    weights = recency_weights(train, float(configuration['half_life']))
    x,width=direct_design(train,configuration,True)
    y=train.chaseWin.to_numpy(float)
    penalty=np.ones(x.shape[1]);penalty[0]=0
    penalty[width:]=float(configuration['era_penalty'])
    alpha=float(configuration['alpha'])
    def objective(c):
        z=x@c
        value=np.mean(weights*(np.logaddexp(0,z)-y*z))+alpha*np.dot(c*penalty,c)/2
        gradient=np.asarray(x.T@(weights*(expit(z)-y)))/len(y)+alpha*c*penalty
        return value,gradient
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1200,'ftol':1e-10,'gtol':1e-7})
    if not fit.success:
        fit=minimize(objective,fit.x,jac=True,method='L-BFGS-B',options={'maxiter':1200,'ftol':1e-10,'gtol':1e-7})
    if not fit.success:raise RuntimeError(fit.message)
    configuration['converged']=True
    return dict(configuration=configuration,fitted=fit.x)


def predict_model(model, frame):
    return expit(direct_design(frame, model["configuration"])[0] @ model["fitted"])
