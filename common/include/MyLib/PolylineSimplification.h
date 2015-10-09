#pragma once

#include <SCI/Vex3.h>

#include <vector>
#include <queue>

using namespace std;

namespace MyLib {


    class Segment {
    public:
            int n0, n1;
            float err;
            int n_err;
            bool cr;
            float t_err;

            Segment( int _i0, int _i1, int _nerr, float _err, bool _cr = false ) : n0(_i0), n1(_i1), err(_err), n_err(_nerr), cr(_cr){ t_err = 0; }

            bool operator < (const Segment & seg) const {
                    if(cr) return n0 < seg.n0;
                    return err < seg.err;
            }
    };



    class PolylineSimplificationRDP : public vector<SCI::Vex3> {

    protected:
        float maximum_error;

        pair< int, float > GetMaxErrorNode( vector<SCI::Vex3> & node, int i0, int i1 );

    public:

        PolylineSimplificationRDP( );

        void Calculate( vector<SCI::Vex3> & node, float errMax, int nodeMax );

    };

/*

class CompressedNodeCatmullRom : public IndexedNode {

protected:
        SCI::Vex3 interpolate(SCI::Vex3 & a, SCI::Vex3 & b, SCI::Vex3 & c, SCI::Vex3 & d, float t );
        void UpdateErrors(   vector<SCI::Vex3> & node, vector< Segment > & segs );
        int  GetMaxErrorSeg( vector<SCI::Vex3> & node, vector< Segment > & segs );

public:

        CompressedNodeCatmullRom( vector<SCI::Vex3> & node, float errMax, int nodeMax );

        virtual SCI::Vex3 GetPosition( float state );


};
*/
}
