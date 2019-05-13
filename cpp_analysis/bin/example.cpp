

#include "TString.h"
#include "friendTreeInjector.h"
#include <iostream>

int main(int argc, char* argv[]){
    if(argc<2) return -1;

    TString infile = argv[1];

    friendTreeInjector intree;
    intree.addFromFile(infile);
    intree.setSourceTreeName("tree");

    intree.createChain();

    auto c = intree.getChain();

    std::cout << c->GetEntries() <<std::endl;

}
