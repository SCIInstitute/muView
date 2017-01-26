#ifndef PROVENANCENODE_H
#define PROVENANCENODE_H


#include <RemoteDimensionalityReduction.h>
#include <Data/BasicData.h>
#include <Drawable/DrawableObject.h>
#include <SCI/Vex2.h>
#include <SCI/Mat4.h>
#include <GL/oglFrameBufferObject.h>
#include <Drawable/ColorLine.h>
#include <Histogram2D.h>
#include <BoundingRegion.h>

#include <map>

namespace Drawable {

    class ProvenanceNode : public QObject {

        Q_OBJECT

    public:
        ProvenanceNode( Data::ProxyData * data_in, SCI::Subset & _elements, int method );


              int                                CountDepth( )                   const  ;
              std::vector<int>                   CountLevelWidth( )              const  ;
              oglWidgets::oglFrameBufferObject & GetDrawingSurface( )                   ;
        const ProvenanceNode                   * GetSelected( SCI::Vex2 loc )  const  ;
              ProvenanceNode                   * GetSelected( SCI::Vex2 loc )         ;
              SCI::Mat4                        GetTransform()                  const  ;
              int                                GetMethod()                     const  ;
        const RemoteDimensionalityReduction          * GetDR()                         const  ;
              bool                               isDone()                        const  ;
              SCI::Subset                             GetFeatureList()                const  ;
              SCI::Subset                             GetElementList()                const  ;

              void                               UpdateDrawingSurface( );
              void                               UpdateBins();

              Histogram2D                      & GetHistogram(){ return histogram; }

              void                               TranslateLocation( float dx, float dy );

              void                               ColorData( unsigned int * col );
              void                               UpdatePointers();

              ColorLine                        & GetLine(){ return line; }

              BBox  GetBBox(){ return BBox(); }


              ProvenanceNode *                AddChild( SCI::Subset & _elements, int method, BBox bbox );

    protected:
        void CountLevelWidth( int level, std::vector<int> & list ) const ;
        void UpdatePointers( std::map< std::pair<int,int>, ProvenanceNode* > map, ProvenanceNode *parent, std::vector<int> &cur_column, int curdepth );
        void BuildMap( std::map< std::pair<int,int>, ProvenanceNode* > & map, std::vector<int> & cur_column, int curdepth  );

    public slots:
        void ComputationCompleted( bool prematureTermination );


    protected:
        // Data for the tree
        ProvenanceNode *                parent;
        ProvenanceNode *                neighbor_left;
        ProvenanceNode *                neighbor_right;
        std::vector< ProvenanceNode * > children;
        std::vector< BBox >             child_bboxes;

        // Input and output data
        Data::ProxyData *               data_in;
        Data::BasicData *               data_out;
        RemoteDimensionalityReduction *       dr;
        SCI::Subset                          features;
        SCI::Subset                          elements;


    protected:

        SCI::Vex2                    location;
        ColorLine                      line;

        oglWidgets::oglFrameBufferObject fbo;
        bool                             fbo_init;
        int                              curdraw;
        SCI::Mat4                      transform;

        SCI::Vex2                      bbmin;
        SCI::Vex2                      bbmax;

        Histogram2D                      histogram;

        int     curbin;

        friend class Provenance;

    };

}


#endif // PROVENANCENODE_H
