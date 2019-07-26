

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <iostream>
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "Math/Vector3D.h"
#include <math.h>
#include "TChain.h"

template<class T>
T deltaPhi( T a, T b ){

    const T pi = 3.14159265358979323846;
    T delta = (a - b);
    while (delta >= pi)  delta-= 2.* pi;
    while (delta < -pi)  delta+= 2.* pi;
    return delta;
}



void mergeOF(TH1D* h){
    double lastb = h->GetBinContent(h->GetNbinsX());
    h->SetBinContent(h->GetNbinsX(), lastb+h->GetBinContent(h->GetNbinsX()+1));
}

float DR(const float eta1, const float phi1,
        const float eta2, const float phi2){

    double deta = eta1-eta2;
    double dphi = deltaPhi<float>(phi1,phi2);

    return sqrt(deta*deta + dphi*dphi);
}

float distance(ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorA,
        ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorB){

    auto dx = RhoEtaPhiVectorA.X() - RhoEtaPhiVectorB.X();
    auto dy = RhoEtaPhiVectorA.Y() - RhoEtaPhiVectorB.Y();
    auto dz = 0.1*(RhoEtaPhiVectorA.Z() - RhoEtaPhiVectorB.Z());

    return sqrt(dx*dx + dy*dy + dz*dz);
}

float distancexy(ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorA,
        ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorB){

    auto dx = RhoEtaPhiVectorA.X() - RhoEtaPhiVectorB.X();
    auto dy = RhoEtaPhiVectorA.Y() - RhoEtaPhiVectorB.Y();

    return sqrt(dx*dx + dy*dy);
}

int main(int argc, char* argv[]){
    if(argc<2) return -1;

    TString infile = argv[1];

    //TFile f(infile, "READ");
    TChain * tree = new TChain();
    std::cout << "adding " << infile+"/Delphes" << std::endl;
    tree->Add(infile+"/Delphes");
    if(!tree || tree->IsZombie()){
        std::cerr << "tree has a problem" << std::endl;
        return -1;
    }


    std::vector<std::vector<float> > * simcluster_features =0;

    tree->SetBranchAddress("simcluster_features",&simcluster_features);

    if(tree->GetEntries()<1){
        std::cerr << "tree has 0 entries" <<std::endl;
        return -2;
    }

    TFile outtfile("plots.root","RECREATE");

    TH2D simcluster_distance("simcluster_distance","simcluster_distance",
            20,0.,0.001,
            20,0.,20);

    TH2D n_close_simclusters("n_close_simclusters","n_close_simclusters",
            20,0.,20,
            20,0.,20);

    TH2D n_closexy_simclusters("n_closexy_simclusters","n_closexy_simclusters",
            20,0.,20,
            20,0.,20);

    double mindistance=.5;

    const int nentries = tree->GetEntries();
    for(int event=0;event<nentries;event++){
        tree->GetEntry(event);

        int nclose=0;
        int nclosexy=0;

        for(size_t i=0;i<simcluster_features->size();i++){
            ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >
            va(simcluster_features->at(i).at(3),simcluster_features->at(i).at(1), simcluster_features->at(i).at(2));

            for(size_t j=i;j<simcluster_features->size();j++){
                if(i==j)continue;
                ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >
                vb(simcluster_features->at(j).at(3),simcluster_features->at(j).at(1), simcluster_features->at(j).at(2));

                float dr=DR(simcluster_features->at(i).at(1), simcluster_features->at(i).at(2),
                        simcluster_features->at(j).at(1), simcluster_features->at(j).at(2));

                float dist = distance(va,vb);

                if(dist<mindistance)nclose++;
                if(distancexy(va,vb)<mindistance)nclosexy++;
                simcluster_distance.Fill(dr,(float)simcluster_features->size());

            }
        }
        n_close_simclusters.Fill(simcluster_features->size(),nclose );
        n_closexy_simclusters.Fill(simcluster_features->size(),nclosexy );

    }

    simcluster_distance.Write();
    simcluster_distance.SetMarkerSize(31);
    n_close_simclusters.Write();
    n_close_simclusters.SetMarkerSize(31);
    n_closexy_simclusters.Write();
    n_closexy_simclusters.SetMarkerSize(31);
    outtfile.Close();

    return 0;
}
