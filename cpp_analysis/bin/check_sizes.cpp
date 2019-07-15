

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <iostream>
#include "TH1D.h"
#include "TCanvas.h"

void mergeOF(TH1D* h){
    double lastb = h->GetBinContent(h->GetNbinsX());
    h->SetBinContent(h->GetNbinsX(), lastb+h->GetBinContent(h->GetNbinsX()+1));
}

int main(int argc, char* argv[]){
    if(argc<2) return -1;

    TString infile = argv[1];

    TFile f(infile, "READ");
    TTree * tree = (TTree*) f.Get("Delphes");
    if(!tree || tree->IsZombie()){
        std::cerr << "tree has a problem" << std::endl;
        return -1;
    }


    std::vector<std::vector<float> > * rh_feat =0, * rh_truth = 0, *lc_feat = 0, * lc_truth=0;

    tree->SetBranchAddress("rechit_features",&rh_feat);
    tree->SetBranchAddress("rechit_simcluster_fractions",&rh_truth);
    tree->SetBranchAddress("layercluster_features",&lc_feat);
    tree->SetBranchAddress("layercluster_simcluster_fractions",&lc_truth);

    if(tree->GetEntries()<1){
        std::cerr << "tree has 0 entries" <<std::endl;
        return -2;
    }

    tree->GetEntry(0);
    if(rh_feat->size()<1 || lc_feat->size()<1){
        std::cerr << "first entry has zero rechits or layer cluster" <<std::endl;
        return -3;
    }
    std::cout << "rechit_features: " << rh_feat->at(0).size() << std::endl;
    std::cout << "layercluster_features: " << lc_feat->at(0).size() << std::endl;

    std::cout << "\ncomputing average/max rechits" << std::endl;

    TH1D nrechits("nrechits","nrechits",20,1000,10000);
    TH1D nlayerclusters("nlayerclusters","nlayerclusters",20,100,2000);

    int max_nrechits=0, max_nlayerclusters=0, max_simclusters=0;
    float avg_nrechits=0, avg_nlayerclusters=0;

    const int nentries = tree->GetEntries();
    for(int event=0;event<nentries;event++){
        tree->GetEntry(event);
        const int nrh = rh_feat->size();
        const int nlc = lc_feat->size();
        int nsc = 0;
        if(rh_truth->size()>0)
            nsc=rh_truth->at(0).size();

        if(max_nrechits<nrh)
            max_nrechits=nrh;
        if(max_nlayerclusters<nlc)
            max_nlayerclusters=nlc;
        if(max_simclusters<nsc)
            max_simclusters=nsc;

        avg_nrechits+=nrh;
        avg_nlayerclusters+=nlc;

        nrechits.Fill(nrh);
        nlayerclusters.Fill(nlc);
    }
    avg_nrechits/=(float)nentries;
    avg_nlayerclusters/=(float)nentries;


    std::cout << "max nrechits: " << max_nrechits << std::endl;
    std::cout << "max nlayerclusters: " << max_nlayerclusters << std::endl;
    std::cout << "max nsimclusters: " << max_simclusters << std::endl;
    std::cout << "average nrechits: " << avg_nrechits << std::endl;
    std::cout << "average nlayerclusters: " << avg_nlayerclusters << std::endl;

    mergeOF(&nrechits);
    mergeOF(&nlayerclusters);


    TCanvas cv;
    nrechits.Draw();
    cv.Print("nrechits.pdf");
    nlayerclusters.Draw();
    cv.Print("nlayerclusters.pdf");


    return 0;
}
