import numpy as np
import matplotlib.pyplot as plt

def profile(target,xvar,bins=10,range=None,uniform=False,moments=True,
            quantiles=np.array([0.25,0.75]),average=False):

    if range is None:
        if type(bins) is not int:
            xmin, xmax = bins.min(), bins.max()
        else:
            xmin, xmax = xvar.min(),xvar.max()
    else:
        xmin, xmax = range
    mask = ( xvar >= xmin ) & ( xvar <= xmax )
    xvar = xvar[mask]
    target = target[mask]
    if type(bins) == int:
        if uniform:
            bins = np.linspace(xmin,xmax,num=bins+1)
        else:
            bins = np.percentile( xvar, np.linspace(0,100.,num=bins+1) )
            bins[0] = xmin
            bins[-1] = xmax
    ibins = np.digitize(xvar,bins)-1
    categories = np.eye(np.max(ibins)+1)[ibins]

    ret = [bins]
    mxvar=xvar.reshape(-1,1) * categories
    if average==True :
        if (sum(categories)!=0).all(): 
            ret = [ np.average(mxvar,weights=categories,axis=0) ]
        else:
            ret = [bins[:-1]]
    if moments:
        mtarget = target.reshape(-1,1) * categories
        weights = categories
        mean = np.average(mtarget,weights=categories,axis=0)
        mean2 = np.average(mtarget**2,weights=categories,axis=0)
        ret.extend( [mean, np.sqrt( mean2 - mean**2)] )
    if quantiles is not None:
        values = []
        for ibin in np.arange(categories.shape[1],dtype=int):
            target_in_bin = target[categories[:,ibin].astype(np.bool)]
            if len(target_in_bin)>0:
                values.append( np.percentile(target_in_bin,quantiles*100.,axis=0).reshape(-1,1) )
            else:
                values.append(np.zeros_like(quantiles).reshape(-1,1))
        ret.append( np.concatenate(values,axis=-1) )
    return tuple(ret)