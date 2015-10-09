#include <MyLib/PolylineSimplification.h>

#include <algorithm>

using namespace MyLib;



pair< int, float > PolylineSimplificationRDP::GetMaxErrorNode( vector<SCI::Vex3> & node, int i0, int i1 ){
    pair<int,float> max_err(i0,0.0f);
	for(int i = i0+1; i < i1; i++){
		float t = (float)(i-i0) / (float)(i1-i0);
                SCI::Vex3 ip = SCI::lerp( node[i0], node[i1], t );
                float err = (ip-node[i]).Length();
		if(err > max_err.second){
			max_err.first = i;
			max_err.second = err;
		}
	}
	return max_err;
}

PolylineSimplificationRDP::PolylineSimplificationRDP( ){
    maximum_error = 0;
}

void PolylineSimplificationRDP::Calculate( vector<SCI::Vex3> & node, float errMax, int nodeMax ) {
    clear();
    maximum_error = 0;
	
    priority_queue< Segment > pq;
    vector< int > indexed_nodes;

    int i0 = 0;
    int i1 = (int)node.size()-1;
    pair<int,float> e = GetMaxErrorNode( node, i0, i1 );
    pq.push( Segment(i0,i1,e.first,e.second) );
    indexed_nodes.push_back( i0 );
    indexed_nodes.push_back( i1 );

    while( pq.size() > 0 && pq.top().err > errMax && (int)indexed_nodes.size() < nodeMax){
            Segment curr = pq.top();
            pq.pop();

            indexed_nodes.push_back( curr.n_err );

            if( curr.n_err-curr.n0 > 1 ){
                    e = GetMaxErrorNode( node, curr.n0, curr.n_err );
                    pq.push( Segment(curr.n0, curr.n_err, e.first, e.second) );
            }
            if( curr.n1-curr.n_err > 1 ){
                    e = GetMaxErrorNode( node, curr.n_err, curr.n1 );
                    pq.push( Segment(curr.n_err, curr.n1, e.first, e.second) );
            }
    }

    maximum_error = ( pq.size() == 0 ) ? 0 : pq.top().err;

    sort( indexed_nodes.begin(), indexed_nodes.end() );

    for(int i = 0; i < (int)indexed_nodes.size(); i++){
            push_back( node[ indexed_nodes[i] ] );
    }

}



/*


void CompressedNodeCatmullRom::UpdateErrors( vector<SCI::Vex3> & node, vector< Segment > & segs ){
        for( int s = 0; s < (int)segs.size(); s++){
		if(segs[s].err == FLT_MAX){
			segs[s].t_err = 0;
			segs[s].err = 0;

			int an = segs[ (s>0) ? (s-1) : s ].n0;
			int bn = segs[ s ].n0;
			int cn = segs[ s ].n1;
                        int dn = segs[ ((s+1)>=(int)segs.size()) ? s : (s+1) ].n1;

                        SCI::Vex3 a = node[ an ];
                        SCI::Vex3 b = node[ bn ];
                        SCI::Vex3 c = node[ cn ];
                        SCI::Vex3 d = node[ dn ];

			for(int j = bn+1; j < cn-1; j++){
				float t = (float)(j-bn) / (float)(cn-bn);
                                SCI::Vex3 p = interpolate( a, b, c, d, t );
                                float err = (p-node[j]).Length();

				segs[s].t_err += err;
				if(err > segs[s].err ){
					segs[s].n_err = j;
					segs[s].err = err;
				}
			}

		}
	}
}

int CompressedNodeCatmullRom::GetMaxErrorSeg( vector<SCI::Vex3> & node, vector< Segment > & segs ){
	int seg = 0;
	maximum_error = segs[0].err;
	average_error = segs[0].err;

        for( int s = 1; s < (int)segs.size(); s++){
		if( segs[s].err > maximum_error ){
			seg = s;
			maximum_error = segs[s].err;
		}
		average_error += segs[s].t_err;
	}
	average_error /= node.size();
	return seg;
}

CompressedNodeCatmullRom::CompressedNodeCatmullRom( vector<SCI::Vex3> & node, float errMax, int nodeMax ) : IndexedNode( node.size() ) {

	vector< Segment > segs;
	segs.push_back( Segment( 0, node.size()-1, -1, FLT_MAX, true ) );
	insert_order.push_back( 0 );
	insert_order.push_back( node.size()-1 );

	UpdateErrors( node, segs );

        while( (int)(segs.size()+1) < nodeMax ){
		int s = GetMaxErrorSeg( node, segs );
		if(segs[s].err < errMax) break;

		insert_order.push_back( segs[s].n_err );

		int n0 = segs[s].n0;
		int n1 = segs[s].n_err;
		int n2 = segs[s].n1;

		segs[s].n0 = n0;
		segs[s].n1 = n1;
		segs[s].err = FLT_MAX;
		if(s > 0){ segs[s-1].err = FLT_MAX; }
                if((s+1) < (int)segs.size()){ segs[s+1].err = FLT_MAX; }
		segs.push_back( Segment( n1, n2, -1, FLT_MAX, true ) );
		sort( segs.begin(), segs.end() );
		UpdateErrors( node, segs );
	}

	indexed_nodes.push_back( IndexedPosition(0,node[0]) );
        for(int i = 0; i < (int)segs.size(); i++){
		indexed_nodes.push_back( IndexedPosition(segs[i].n1,node[segs[i].n1]) );
	}
}

SCI::Vex3 CompressedNodeCatmullRom::interpolate(SCI::Vex3 & a, SCI::Vex3 & b, SCI::Vex3 & c, SCI::Vex3 & d, float t ){
	float q0 = t * ( (2.0f-t)*t - 1.0f);
	float q1 = t * t * (3.0f*t-5.0f) + 2.0f;
	float q2 = t * ( (4.0f-3.0f*t)*t + 1.0f);
	float q3 = t * t * (t-1.0f);
	return (a*q0+b*q1+c*q2+d*q3)*0.5f;
}

SCI::Vex3 CompressedNodeCatmullRom::GetPosition( float state ){
        for(int i = 0; i < (int)indexed_nodes.size()-1; i++){
		if( indexed_nodes[i].first <= state && state <= indexed_nodes[i+1].first){

			IndexedPosition a = indexed_nodes[ (i>0) ? (i-1) : i ];
			IndexedPosition b = indexed_nodes[ i ];
			IndexedPosition c = indexed_nodes[ i+1 ];
                        IndexedPosition d = indexed_nodes[ ((i+2)>=(int)indexed_nodes.size()) ? (i+1) : (i+2) ];

			float t = (state-(float)b.first) / (float)(c.first-b.first);

			return interpolate(a.second,b.second,c.second,d.second,t);
		}
	}
        return SCI::Vex3(0,0,0);
}


*/

