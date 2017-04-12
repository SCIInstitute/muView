#ifndef PCAAXIS_H
#define PCAAXIS_H

#include <Data/Mesh/PointMesh.h>
#include <Data/Mesh/TriMesh.h>
#include <Data/PointData.h>
#include <Data/Mesh/SolidMesh.h>

#include <DimensionalityReduction.h>



class PCAaxis
{

public:
    PCAaxis();
    
    void SetData(Data::PointData * pdata, Data::Mesh::PointMesh *pmesh, Data::Mesh::SolidMesh *tdata, Data::Mesh::ColorMesh     * colormap );



public:
    DimensionalityReduction* getDMR();

    double * getPrincipalComponent(int component);
    int getSubsetSize();
    void getSubsetIndices(std::vector<int>& indexList);

protected:

public:

    Data::PointData       * pdata;
    Data::Mesh::PointMesh * pmesh;
    Data::Mesh::SolidMesh * tdata;
    Data::Mesh::ColorMesh * colormap;

    DimensionalityReduction dmr;

    Data::DenseMultiDimensionalData pca_out;
    SCI::Subset full_set;
    SCI::Subset features;

    int subsetSize;

};

#endif // PCAAXIS_H
