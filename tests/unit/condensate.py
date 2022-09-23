
from assign_condensate_op import calc_ragged_shower_indices, BuildAndAssignCondensatesBinned, calc_ragged_cond_indices
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from RaggedLayers import RaggedDense, RaggedMixCPAndHitInfo

def make_data(nx, ny, nrs):
    
    nx, ny = (nx, ny)
    def oners():
        x = np.linspace(0, 6, nx)
        y = np.linspace(0, 6, ny)
        xv, yv = np.meshgrid(x, y)
        
        x = np.concatenate([xv[...,np.newaxis], yv[...,np.newaxis]],axis=-1)
        x = np.reshape(x, [nx*ny,2])
        beta = np.random.rand(nx*ny,1)
        
        return x, beta, nx*ny
    
    xs=[]
    betas=[]
    rs=[0]
    for _ in range(nrs):
        x,b,r = oners()
        xs.append(x)
        betas.append(b)
        rs.append(r)
        
    rs = np.cumsum(np.array(rs),axis=0)
    xs = np.concatenate(xs, axis=0)
    betas = np.concatenate(betas, axis=0)
    
    return tf.constant(xs, dtype='float32'), tf.constant(betas, dtype='float32'), tf.constant(rs, dtype='int32')


def test_indices(betathresh):
    x,b,rs = make_data(690,30,5)
    
    nocond = tf.where( b <0.1, 1, tf.zeros_like(b))
    
    
    assignment, asso, alphaidx, _, ncond = BuildAndAssignCondensatesBinned(x,
                            b,
                            tf.ones_like(b),
                            rs,
                            0.21,
                            no_condensation_mask = nocond,
                            assign_by_max_beta=False)
    
    
    
    #exit()
    # ncondensates does not include noise, but then it is used as if it would
    # FIXME recalc on the fly using max and min assignment index per row split FIXME!
    irdxs = calc_ragged_shower_indices(assignment, rs)
    
    #print('indices A done>>>', ncond, assignment)
    rcidx, retrev, flatrev = calc_ragged_cond_indices(assignment, alphaidx, rs)
    
    #sanity check
    rassignment = tf.gather_nd(assignment, irdxs)
    rassignment = tf.reduce_min(rassignment, axis=2)#same
    #rassignment = tf.ragged.boolean_mask(rassignment, rassignment < 2147483647)#remove empty
    
    
    cprassignement = tf.gather_nd(assignment, rcidx)
    #first check
    #print('>>first', rassignment, '\n',cprassignement)
    tf.assert_equal((rassignment-cprassignement).values, 0) # == for ragged sometimes creates issues
    
    #back from ragged
    retass = tf.gather_nd(cprassignement, retrev)
    #print('second >> ',retass, '\n',assignment)
    tf.assert_equal(retass, assignment)
    
    #back from flat
    #print('third a >> ',cprassignement, assignment, '\n',retrev, flatrev, ncond)
    retass = tf.gather_nd(cprassignement.values, flatrev)
    #print('third >> ',retass, '\n',assignment)
    tf.assert_equal(retass, assignment)

for bt in 10*[0.001, 0.1, 0.2, 0.4]:
    print(bt)
    test_indices(bt)

exit()
#print('rcidx', rcidx)
#exit()
xcp = tf.gather_nd(x, rcidx)
xcpass = tf.gather_nd(assignment, rcidx)

#print('xcpass',xcpass)
exit()

xr = tf.gather_nd(x, irdxs)

print('xr',xr)
print('xcp',xcp)

xmixed = RaggedMixCPAndHitInfo(operation='add')([xr,xcp])
print('xmixed',xmixed)
#exit()

xr = tf.expand_dims(xr, axis=2)
print('xr',xr.shape)
#exit()
xr = RaggedDense(2,activation='elu')(xr)
print(xr, xr.shape)
#exit()

xr = tf.gather_nd(x, irdxs)
xmr = tf.reduce_mean(xr, axis=2)

'''

make ragged condensation point select indices:
event x CPs




'''

mrs = xmr.row_splits
#print(xr, xmr, mrs)

for i in range(len(rs) - 1):
    
    print('..',xmr[i])
    
    plt.scatter(x[rs[i]:rs[i+1]][:,0],x[rs[i]:rs[i+1]][:,1], c = assignment[rs[i]:rs[i+1]])
    
    plt.scatter(xmr[i,:,0] , xmr[i,:,1] , marker='x')
    plt.show()
    plt.close()