import uproot3 as uproot
from uproot3_methods import TLorentzVector,TLorentzVectorArray
from numba import jit
import awkward0 as ak
import numpy as np

def unmergedIndices(mergedSimClusterIdx):
    # Keeping track of the indices in case they're needed somewhere else later
    groups = []
    entries = []
    nev = len(mergedSimClusterIdx)
    for i in range(nev):
        nsc = len(mergedSimClusterIdx)
        for j in range(nsc):
            vals  = mergedSimClusterIdx[i] == j
            matches = np.nonzero(vals)[0]
            entries.append(matches)
        print(entries)
        print(np.array(entries, dtype='int32'))
        groups.append(entries)

    # Not used currently, but could be useful, so leaving here
    return ak.JaggedArray.fromiter(groups)

f = uproot.open("/eos/cms/store/cmst3/group/hgcal/CMG_studies/kelong/GeantTruthStudy/SimClusterNtuples/testNanoML.root")
tree = f["Events"]

pt = tree["SimCluster_pt"].array()
eta = tree["SimCluster_eta"].array()
phi = tree["SimCluster_phi"].array()
m = tree["SimCluster_mass"].array()

unzeros = ak.JaggedArray.fromoffsets(pt.offsets, np.zeros(len(pt.flatten()))) 

unmerged = TLorentzVectorArray.from_ptetaphim(pt,eta,phi,m)

pt = tree["MergedSimCluster_pt"].array()
eta = tree["MergedSimCluster_eta"].array()
phi = tree["MergedSimCluster_phi"].array()
m = tree["MergedSimCluster_mass"].array()

merged = TLorentzVectorArray.from_ptetaphim(pt,eta,phi,m)

mergedSimClusterIdx = tree["SimCluster_MergedSimClusterIdx"].array()

unmergedIds = tree["SimCluster_pdgId"].array()
muon_filt = np.abs(unmergedIds == 13)

flat_filt = muon_filt.flatten()

muonsc_pts = unzeros.flatten()
muonsc_pts[flat_filt] = unmerged.pt.flatten()[flat_filt]
muonsc_pts = ak.JaggedArray.fromoffsets(unzeros.offsets, muonsc_pts)

muonsc_etas = unzeros.flatten()
muonsc_etas[flat_filt] = unmerged.eta.flatten()[flat_filt]
muonsc_etas = ak.JaggedArray.fromoffsets(unzeros.offsets, muonsc_etas)

muonsc_phis = unzeros.flatten()
muonsc_phis[flat_filt] = unmerged.phi.flatten()[flat_filt]
muonsc_phis = ak.JaggedArray.fromoffsets(unzeros.offsets, muonsc_phis)

muonsc_mass = unzeros.flatten()
muonsc_mass[flat_filt] = unmerged.mass.flatten()[flat_filt]
muonsc_mass = ak.JaggedArray.fromoffsets(unzeros.offsets, muonsc_mass)

muons = TLorentzVectorArray.from_ptetaphim(muonsc_pts, muonsc_etas, muonsc_phis, muonsc_mass)

@jit(nopython=True)
def correctE(merged, muons, mergedSimClusterIdx):
    newE = list()
    for e in range(len(merged)):
        eventE = list()
        for i,m in enumerate(merged[e]):
            musum = muons[e][mergedSimClusterIdx[e] == i].sum()
            eventE.append((m-musum).energy)
        newE.append(eventE)

    correctedE = ak.JaggedArray.fromiter(newE)
    return correctedE


corrE = correctE(merged, muons, mergedSimClusterIdx)
