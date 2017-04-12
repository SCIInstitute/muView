#include <muView/PCAaxis.h>

PCAaxis::PCAaxis(){
    pdata = 0;
    pmesh = 0;
    tdata = 0;

    subsetSize = 100;

}



DimensionalityReduction* PCAaxis::getDMR(){
    return &dmr;
}


void PCAaxis::getSubsetIndices(std::vector<int>& indexList){
    for (int i=0;i<features.size(); i++){
        indexList.push_back(features.at(i));
        std::cout << features.at(i)<<" "<<std::endl;
    }
}

double* PCAaxis::getPrincipalComponent(int component){
    return dmr.GetPrincipalComponent (component);;
}

int PCAaxis::getSubsetSize() {
    return subsetSize;
}

void PCAaxis::SetData(Data::PointData * _pdata, Data::Mesh::PointMesh * _pmesh, Data::Mesh::SolidMesh * _tdata, Data::Mesh::ColorMesh * _colormap){

    pdata    = _pdata;
    pmesh    = _pmesh;
    tdata    = _tdata;
    colormap = _colormap;

    std::cout << "PCAaxis: " << dmr.GetMethodCount() << " methods" << std::endl;
    for(int i = 0; i < dmr.GetMethodCount(); i++){
        std::cout << "PCAaxis: " << dmr.GetMethodName(i) << std::endl;
    }

    //pdata->GetElementCount();

    std::cout << "PCAaxis: set size but need to change strat method and how data vectors are set: " << std::endl;

    full_set = SCI::Subset( pdata->GetElementCount() );//number of vertices
    features.GetRandomSubset(full_set, subsetSize);//not on all vertices

    //todo reduce to random number
    // doesn't matter because we want transformation matrix
    // make sure we get enough axis to do so
    pca_out.Resize( pdata->GetDimension(), 3);//elem, dim


    std::cout << "PCAaxis: data dimensions num run " << pdata->GetDimension() << std::endl;
    std::cout << "PCAaxis: data num vertices " << pdata->GetVoxelCount() << std::endl;


    std::cout << "PCAaxis: data in dimension " << pdata->GetDimension() << std::endl;
    std::cout << "PCAaxis: data out dimension " << pca_out.GetDimension() << std::endl;

    std::cout << "PCAaxis: full_set size " << full_set.size() << std::endl;
    std::cout << "PCAaxis: feature size" << features.size() << std::endl;

    std::cout << "PCAaxis: Starting PCA with " << features.size() << " features" << std::endl;
    std::cout << "PCAaxis: start computation " << std::endl;

    //todo make pdata smaller/ or put in feature subset as pdata otherwise too much data
    //todo pdata needs to be smaller.
    // or getDimensions in StartAxis needs to take rand subset and only those
    dmr.StartAxis(0, *pdata, features, pca_out, full_set );
    std::cout << "PCAaxis: Finished PCA feature phase" << std::endl;
}
