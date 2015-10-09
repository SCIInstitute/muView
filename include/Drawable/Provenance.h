#ifndef PROVENANCE_H
#define PROVENANCE_H

#include <Drawable/ProvenanceNode.h>
#include <QAction>

namespace Drawable {

    class Provenance : public Drawable::DrawableObject {

        Q_OBJECT

    public:
        Provenance();

        virtual void Draw( );

        void Initialize( Data::ProxyData & data );

        void Push( BBox bbox );
        void Push( int dr_method );

        ProvenanceNode * GetCurrentNode();

        virtual bool MouseClick(int button, int state, int x, int y);
        virtual bool Keypress( int key, int x, int y );
        virtual bool ClearSelection( );

    public slots:
        void Push( QAction * dr_method );

    protected:

        Data::ProxyData * data_in;
        ProvenanceNode  * head;
        ProvenanceNode  * curr;

        bool selected;

        void Push( BBox bbox, SCI::Subset newelems, int dr_method );
        void CalculateNodeLocation( ProvenanceNode * node, std::vector<int> & cur_column, std::vector<int> level_widths, int curdepth, int maxdepth );
        void CalculateNodeLocationPass2( ProvenanceNode * node );
        void CalculateNodeLocationPass3( ProvenanceNode * node );
        void DrawNode( ProvenanceNode * node );
    };


}


#endif // PROVENANCE_H
