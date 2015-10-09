#include <Data/Mesh/TriMesh.h>

#include <vector>
#include <algorithm>
#include <iostream>

#include <GL/oglCommon.h>

#include <omp.h>



using namespace Data::Mesh;

Triangle::Triangle( FILE * file, bool binary ){
    if(binary){
        fread( data, sizeof(int), 3, file );
    }
    else{
        fscanf( file, " %i %i %i ", &v0, &v1, &v2 );
    }
}

Triangle::Triangle( int _v0, int _v1, int _v2 ) : v0(_v0), v1(_v1), v2(_v2) { }

bool Triangle::operator < ( const Triangle right ) const {
    if( SCI::Min( v0, v1, v2 ) < SCI::Min( right.v0, right.v1, right.v2 ) ){ return true; }
    if( SCI::Min( v0, v1, v2 ) > SCI::Min( right.v0, right.v1, right.v2 ) ){ return false; }
    if( SCI::Mid( v0, v1, v2 ) < SCI::Mid( right.v0, right.v1, right.v2 ) ){ return true; }
    if( SCI::Mid( v0, v1, v2 ) > SCI::Mid( right.v0, right.v1, right.v2 ) ){ return false; }
    if( SCI::Max( v0, v1, v2 ) < SCI::Max( right.v0, right.v1, right.v2 ) ){ return true; }
    return false;
}

bool Triangle::operator != ( const Triangle right ) const {
    if( SCI::Min( v0, v1, v2 ) != SCI::Min( right.v0, right.v1, right.v2 ) ){ return true; }
    if( SCI::Mid( v0, v1, v2 ) != SCI::Mid( right.v0, right.v1, right.v2 ) ){ return true; }
    if( SCI::Max( v0, v1, v2 ) != SCI::Max( right.v0, right.v1, right.v2 ) ){ return true; }
    return false;
}

bool Triangle::operator == ( const Triangle right ) const {
    if( SCI::Min( v0, v1, v2 ) != SCI::Min( right.v0, right.v1, right.v2 ) ){ return false; }
    if( SCI::Mid( v0, v1, v2 ) != SCI::Mid( right.v0, right.v1, right.v2 ) ){ return false; }
    if( SCI::Max( v0, v1, v2 ) != SCI::Max( right.v0, right.v1, right.v2 ) ){ return false; }
    return true;
}

void Triangle::Write( FILE * outfile ){
    fwrite( data, sizeof(int), 3, outfile );
}

float Triangle::GetArea( const PointMesh &pmesh ){
    SCI::Vex3 v = pmesh.GetVertex( v1 ) - pmesh.GetVertex( v0 );
    SCI::Vex3 w = pmesh.GetVertex( v2 ) - pmesh.GetVertex( v0 );
    return SCI::cross( v, w ).Magnitude();
}

SCI::Vex3 Triangle::GetNormal( const PointMesh & pmesh ){
    return SCI::cross( pmesh.GetVertex( v1 ) - pmesh.GetVertex( v0 ), pmesh.GetVertex( v2 ) - pmesh.GetVertex( v0 ) ).UnitVector();
}

bool Triangle::isDegenerate(){
    return (v0==v1) || (v0==v2) || (v1==v2);
}

void Triangle::Flip(){
    std::swap( v1, v2 );
}

Edge Triangle::GetEdge(int i){
    if(i == 0){ return Edge(v0,v1); }
    if(i == 1){ return Edge(v1,v2); }
    if(i == 2){ return Edge(v2,v0); }
    return Edge();
}

float Triangle::GetDistance( SCI::Vex3 pnt, const PointMesh &pmesh, bool quick_estimate ){

    // get vertices
    SCI::Vex3 p0 = pmesh.GetVertex( v0 );
    SCI::Vex3 p1 = pmesh.GetVertex( v1 );
    SCI::Vex3 p2 = pmesh.GetVertex( v2 );

    // quick version just returns the smallest distance to a vertex
    if( quick_estimate ){
        float d = FLT_MAX;
        d = SCI::Min( d, (pnt-p0).Length() );
        d = SCI::Min( d, (pnt-p1).Length() );
        d = SCI::Min( d, (pnt-p2).Length() );
        return d;
    }

    // Find the triangle normal
    SCI::Vex3 n  = SCI::cross( p2-p0, p1-p0 ).UnitVector();

    // Find the closest point on the plane
    SCI::Vex3 p  = pnt - n * SCI::dot( pnt - p0, n );

    // Find perpendiculars on the plane to each edge
    SCI::Vex3 l0 = SCI::cross( p1 - p0, n ).UnitVector();
    SCI::Vex3 l1 = SCI::cross( p2 - p1, n ).UnitVector();
    SCI::Vex3 l2 = SCI::cross( p0 - p2, n ).UnitVector();

    // If point is outside of the triangle, project it to the appropriate edge
    if( SCI::dot( p-p0, l0 ) * SCI::dot( p2-p0, l0 ) < 0 ){ p = p - l0 * SCI::dot( p-p0, l0 ); }
    if( SCI::dot( p-p1, l1 ) * SCI::dot( p0-p1, l1 ) < 0 ){ p = p - l1 * SCI::dot( p-p1, l1 ); }
    if( SCI::dot( p-p2, l2 ) * SCI::dot( p1-p2, l2 ) < 0 ){ p = p - l2 * SCI::dot( p-p2, l2 ); }

    // Distance between closest point inside triangle and the input point
    return ( pnt - p ).Length();
}

SCI::BoundingBox Triangle::GetBoundingBox( const PointMesh & pmesh ){
    SCI::BoundingBox bb;
    bb.Expand( pmesh.GetVertex(v0) );
    bb.Expand( pmesh.GetVertex(v1) );
    bb.Expand( pmesh.GetVertex(v2) );
    return bb;
}

void Triangle::ExtractIsoline( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const std::vector<float> & vals, float iso_val ){
    float d0 = vals[v0];
    float d1 = vals[v1];
    float d2 = vals[v2];

    float t0 = (iso_val-d0)/(d1-d0);
    float t1 = (iso_val-d1)/(d2-d1);
    float t2 = (iso_val-d2)/(d0-d2);

    int cnt = 0;
    if( t0 > 0 && t0 < 1 ){ cnt++; verts.points.push_back( SCI::lerp(verts_in.points[v0], verts_in.points[v1], t0 ) ); }
    if( t1 > 0 && t1 < 1 ){ cnt++; verts.points.push_back( SCI::lerp(verts_in.points[v1], verts_in.points[v2], t1 ) ); }
    if( t2 > 0 && t2 < 1 ){ cnt++; verts.points.push_back( SCI::lerp(verts_in.points[v2], verts_in.points[v0], t2 ) ); }

    if(cnt == 2){
        edges.push_back( Edge(verts.points.size()-2, verts.points.size()-1) );
    }
}







TriMesh::TriMesh(){

}

void TriMesh::BuildSpacePartition( const PointMesh & verts ){
    if( kdtree.isBuilt() ) return;
    printf("TriMesh: building kdtree\n"); fflush(stdout);
    for(int i = 0; i < (int)size(); i++){
        kdtree.InsertElement( i, at(i).GetBoundingBox( verts ) );
    }
}


void TriMesh::SortByPainters(std::vector<int> & render_order, SCI::Mat4 proj, const Data::Mesh::PointMesh & pnts){

    int i;

    // Calculate Projected Location of Each Point
    std::vector<float> dpt( pnts.points.size() );
    #pragma omp parallel shared(dpt,pnts) private(i)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for( i = 0; i < (int)pnts.points.size(); i++ ){
            dpt[i] = (proj * pnts.points.at(i)).z;
        }
    }

    // Calculate Approximate Projected Depth of Each Triangle
    std::vector< std::pair<float, int> > dpt_list( size() );
    #pragma omp parallel shared(dpt_list,dpt) private(i)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for( i = 0; i < (int)size(); i++){
            float cdpt = 0;
            cdpt += dpt.at( at(i).v0 );
            cdpt += dpt.at( at(i).v1 );
            cdpt += dpt.at( at(i).v2 );
            dpt_list[i] = std::make_pair( cdpt, i );
        }
    }

    // Sort Triangle By Depth
    std::sort( dpt_list.begin(), dpt_list.end() );

    // Set render order
    render_order.resize( size() );
    #pragma omp parallel shared(render_order,dpt_list) private(i)
    {
        #pragma omp for schedule( dynamic, 1000 )
        for(i = 0; i < (int)dpt_list.size(); i++){
            render_order[i] = dpt_list[dpt_list.size()-i-1].second;
        }
    }

}

void TriMesh::clear(){
    std::vector<Triangle>::clear();
    kdtree.Clear();
}

void TriMesh::RemoveSharedFaces( ){

    std::cout << "TriMesh: removing internal triangle faces" << std::endl << std::flush;

    if( size() == 0 ) return;
    if( size() == 1 ) return;

    std::vector< Triangle > tmp_verts( begin(), end() );
    std::sort( tmp_verts.begin(), tmp_verts.end() );
    clear();

    if( tmp_verts[0] != tmp_verts[1]){
        push_back( tmp_verts[0] );
    }
    for(int i = 1; i < (int)tmp_verts.size()-1; i++){
        if( tmp_verts[i-1] != tmp_verts[i] && tmp_verts[i] != tmp_verts[i+1] ){
            push_back( tmp_verts[i] );
        }
    }
    if( tmp_verts[tmp_verts.size()-2] != tmp_verts[tmp_verts.size()-1]){
        push_back( tmp_verts[tmp_verts.size()-1] );
    }

}



SCI::Vex2 TriMesh::GetMinMaxArea( const PointMesh &pmesh ){
    float minv =  FLT_MAX;
    float maxv = -FLT_MAX;
    for( int t = 0; t < (int)size(); t++){
        float a = at(t).GetArea( pmesh );
        minv = SCI::Min( minv, a );
        maxv = SCI::Max( maxv, a );
    }
    std::cout << "TriMesh: found min/max area [" << minv << ", " << maxv << "]" << std::endl << std::flush;
    return SCI::Vex2(minv,maxv);
}

void TriMesh::Draw( const PointMesh &pmesh, const ColorMesh & color ) const {
    glBegin(GL_TRIANGLES);
    if(color.isColorPerElement()){
        for( int t = 0; t < (int)size(); t++){
            glColor4ubv( (GLubyte*)&color.GetColor( t ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v0 ).data );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v1 ).data );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v2 ).data );
        }
    }
    else{
        for( int t = 0; t < (int)size(); t++){
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v0 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v0 ).data );
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v1 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v1 ).data );
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v2 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v2 ).data );
        }
    }
    glEnd();
}

void TriMesh::Draw(const PointMesh & pmesh , SCI::Vex4 color) const {
    glColor4fv( color.data );
    glBegin(GL_TRIANGLES);
    for( int t = 0; t < (int)size(); t++){
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v0 ).data );
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v1 ).data );
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v2 ).data );
    }
    glEnd();
}

void TriMesh::Draw(const std::vector<int> & render_order, const PointMesh &pmesh, const ColorMesh & color ) const {
    glBegin(GL_TRIANGLES);
    if(color.isColorPerElement()){
        for(int i = 0; i < (int)render_order.size(); i++){
            int t = render_order[i];
            glColor4ubv( (GLubyte*)&color.GetColor( t ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v0 ).data );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v1 ).data );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v2 ).data );
        }
    }
    else{
        for(int i = 0; i < (int)render_order.size(); i++){
            int t = render_order[i];
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v0 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v0 ).data );
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v1 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v1 ).data );
            glColor4ubv( (GLubyte*)&color.GetColor( at(t).v2 ) );
            glVertex3fv( (float*)   pmesh.points.at( at(t).v2 ).data );
        }
    }
    glEnd();
}

void TriMesh::Draw(const std::vector<int> & render_order, const PointMesh &pmesh, SCI::Vex4 color ) const {
    glColor4fv( color.data );
    glBegin(GL_TRIANGLES);
    for(int i = 0; i < (int)render_order.size(); i++){
        int t = render_order[i];
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v0 ).data );
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v1 ).data );
        glVertex3fv( (float*)pmesh.GetVertex( at(t).v2 ).data );
    }
    glEnd();
}


void TriMesh::Append( std::vector<Triangle> & new_tris ){
    for(int i = 0; i < (int)new_tris.size(); i++){
        push_back( new_tris[i] );
    }
}

void TriMesh::Subdivide( float max_area, PointMesh &pmesh ){
    std::cout << "TriMesh: subdividing" << std::endl << std::flush;
    int tsize = (int)size();
    for( int t = 0; t < tsize; t++){
        float a = at(t).GetArea( pmesh );
        if( a > max_area ){
            int ind0 = at(t).v0;
            int ind1 = at(t).v1;
            int ind2 = at(t).v2;
            int ind3 = (int)pmesh.points.size();

            at(t) = Triangle( ind0, ind1, ind3 );
            push_back( Triangle( ind1, ind2, ind3 ) );
            push_back( Triangle( ind2, ind0, ind3 ) );

            SCI::Vex4 col0, col1, col2;

            //col0.SetFromUInt( pmesh.colors[ind0] );
            //col1.SetFromUInt( pmesh.colors[ind1] );
            //col2.SetFromUInt( pmesh.colors[ind2] );

            pmesh.points.push_back( ( pmesh.points[ind0] + pmesh.points[ind1] + pmesh.points[ind2] ) / 3.0f );
            //pmesh.colors.push_back( ( ( col0+col1+col2) / 3.0f ).UIntColor() );
        }
    }
    std::cout << "TriMesh: " << tsize << " triangles, now " << size() << " triangles" << std::endl << std::flush;

}

void TriMesh::Append( std::vector<int> & new_indices ){
    for(int i = 0; i < (int)new_indices.size(); i+=3){
        push_back( Triangle( new_indices[i], new_indices[i+1], new_indices[i+2] ) );
    }
}


void TriMesh::FuseCoindidentVertices( PointMesh & pmesh, int range_min, int range_max, float eps ){

    std::map< int, int > idx_map;
    pmesh.FuseCoindidentVertices( idx_map, range_min, range_max, eps );

    std::vector< Triangle > new_tris;
    std::cout << "TriMesh: updating mapping and removing degenerate triangles" << std::endl << std::flush;
    for(int i = 0; i < (int)size(); i++){
        Triangle newt( idx_map[ at(i).v0 ], idx_map[ at(i).v1 ], idx_map[ at(i).v2 ] );
        if( !newt.isDegenerate() ){
            new_tris.push_back( newt );
        }
    }

    std::cout << "TriMesh: " << size() << " triangles reduced to " << new_tris.size() << " triangles" << std::endl << std::flush;

    //tris = new_tris;
    clear();
    assign( new_tris.begin(), new_tris.end() );
}

void TriMesh::FixNormals( const PointMesh & pmesh ){
    std::cout << "TriMesh: fixing face normals (" << size() << " faces)" << std::endl << std::flush;

    std::vector< SCI::Vex3 > normals( pmesh.points.size(), SCI::Vex3() );

    std::cout << "TriMesh: finding point normals" << std::endl << std::flush;
    for(int i = 0; i < (int)size(); i++){
        SCI::Vex3 n = at(i).GetNormal( pmesh );
        normals[ at(i).v0 ] += n;
        normals[ at(i).v1 ] += n;
        normals[ at(i).v2 ] += n;
    }

    for(int i = 0; i < (int)normals.size(); i++){
        normals[i].Normalize();
    }

    float eps = 0.1f;
    int fcount = 0;
    for(int i = 0; i < (int)size(); i++){
        SCI::Vex3 n = at(i).GetNormal( pmesh );
        float d0 = SCI::dot( normals[ at(i).v0 ], n );
        float d1 = SCI::dot( normals[ at(i).v1 ], n );
        float d2 = SCI::dot( normals[ at(i).v2 ], n );

        // see if we should flip normal
        if( ( d0 < eps && (d1 < eps || d2 < eps) ) || (d1 < eps && d2 < eps) ){
            fcount++;
            at(i).Flip();
        }
    }

    std::cout << "TriMesh: flipped " << fcount << " triangles" << std::endl << std::flush;
}

/*
void TriMesh::CurvatureShading( PointMesh & pmesh ){

    std::cout << "TriMesh: curvature shading (" << size() << " faces)" << std::endl << std::flush;

    std::vector< std::pair< int, SCI::Vex3 > > normal_map;

    std::cout << "TriMesh: finding face normals" << std::endl << std::flush;
    for(int i = 0; i < (int)size(); i++){
        SCI::Vex3 n = at(i).GetNormal( pmesh );
        normal_map.push_back( std::make_pair( at(i).v0, n ) );
        normal_map.push_back( std::make_pair( at(i).v1, n ) );
        normal_map.push_back( std::make_pair( at(i).v2, n ) );
    }

    std::cout << "TriMesh: sorting normal map - " << normal_map.size() << " components" << std::endl << std::flush;
    std::sort( normal_map.begin(), normal_map.end() );

    std::cout << "TriMesh: computing shading" << std::endl << std::flush;
    int srt = 0;
    int end = 0;
    for(int i = 1; i <= (int)normal_map.size(); i++){

        //if( i > 100 ) break;

        if( i == normal_map.size() || normal_map[i].first != normal_map[srt].first ){
            end = i-1;

            float t_diff = 0;
            int cnt = 0;
            for(int j = srt; j < end; j++){
                for(int k = j+1; k <= end; k++){
                    t_diff += 1.0f - fabsf( SCI::dot( normal_map[j].second, normal_map[k].second ) );
                    cnt++;
                }
            }

            SCI::Vex4 col;

            col.SetFromUInt( pmesh.colors[ normal_map[srt].first ] );
            SCI::Vex3 tcol = SCI::lerp( col.xyz(), SCI::Vex3(0,0,0), t_diff / (float)cnt );
            pmesh.colors[ normal_map[srt].first ] = SCI::Vex4( tcol, col.a ).UIntColor();

            //col = SCI::Vex4( (normal_map[srt].second + 1.0f) / 2.0f, col.a );
            //pmesh.colors[ normal_map[srt].first ] = col.UIntColor();

            srt = i;
        }

    }
}
*/

void TriMesh::ExtractEdges( EdgeMesh & edges ){
    for(int i = 0; i < (int)size(); i++){
        for(int j = 0; j < 3; j++){
            edges.push_back( at(i).GetEdge(j) );
        }
    }
    edges.RemoveDuplicates();
}

void TriMesh::ColorEdges( const PointMesh &pmesh, const EdgeMesh &edges, ColorMesh & cm ){
    std::cout << "TriMesh: Coloring Edges" << std::endl << std::flush;

    std::cout << "TriMesh: Finding Edge Normals" << std::endl << std::flush;
    std::map< Edge, SCI::Vex3 > norm_map;
    for(int i = 0; i < (int)edges.size(); i++){
        norm_map[edges[i]] = SCI::Vex3(0,0,0);
    }
    for(int i = 0; i < (int)size(); i++){
        SCI::Vex3 norm = at(i).GetNormal(pmesh);
        for(int j = 0; j < 3; j++){
            Edge edge = at(i).GetEdge(j);
            norm_map[edge] += norm;
        }
    }

    std::cout << "TriMesh: Finding Edge Intensities and Colors" << std::endl << std::flush;
    std::map< Edge, float > ints_map;
    for(int i = 0; i < (int)size(); i++){
        SCI::Vex3 norm = at(i).GetNormal(pmesh);
        for(int j = 0; j < 3; j++){
            Edge edge = at(i).GetEdge(j);
            ints_map[edge] = fabsf( SCI::dot( norm_map[edge], norm ) );
        }
    }
    cm.SetColorPerElement();
    for(int i = 0; i < (int)edges.size(); i++){
        cm.push_back( SCI::lerp( SCI::Vex4(0,0,0,1), SCI::Vex4(1,0,0,1), ints_map[edges[i]] ).UIntColor() );
        //cm.push_back( SCI::Vex4(0,0,0,1).UIntColor() );
    }
}


void TriMesh::ExtractIsolineByDataDimension( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const Data::MultiDimensionalData & pdata_in, int dim, float iso_step ){
    std::vector<float> vals;
    SCI::BoundingRange br;
    for(int i = 0; i < pdata_in.GetElementCount(); i++){
        vals.push_back( pdata_in.GetElement(i, dim) );
        br.Expand( vals.back() );
    }
    float isoval = floorf(br.bbmin/iso_step)*iso_step;
    while(isoval < br.bbmax){
        for(int i = 0; i < size(); i++){
            at(i).ExtractIsoline( edges, verts, verts_in, vals, isoval );
        }
        isoval += iso_step;
    }
}

void TriMesh::ExtractIsolineByMeanValue( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const Data::MultiDimensionalData & pdata_in, float iso_step ){
    std::vector<float> vals;
    SCI::BoundingRange br;
    for(int i = 0; i < pdata_in.GetElementCount(); i++){
        vals.push_back( pdata_in.GetElement(i).Mean() );
        br.Expand( vals.back() );
    }
    float isoval = floorf(br.bbmin/iso_step)*iso_step;
    while(isoval < br.bbmax){
        for(int i = 0; i < size(); i++){
            at(i).ExtractIsoline( edges, verts, verts_in, vals, isoval );
        }
        isoval += iso_step;
    }
}

void TriMesh::ExtractIsolineByStdevValue( EdgeMesh & edges, PointMesh & verts, const PointMesh & verts_in, const Data::MultiDimensionalData & pdata_in, float iso_step ){
    std::vector<float> vals;
    SCI::BoundingRange br;
    for(int i = 0; i < pdata_in.GetElementCount(); i++){
        vals.push_back( pdata_in.GetElement(i).Stdev() );
        br.Expand( vals.back() );
    }
    float isoval = floorf(br.bbmin/iso_step)*iso_step;
    while(isoval < br.bbmax){
        for(int i = 0; i < size(); i++){
            at(i).ExtractIsoline( edges, verts, verts_in, vals, isoval );
        }
        isoval += iso_step;
    }
}
