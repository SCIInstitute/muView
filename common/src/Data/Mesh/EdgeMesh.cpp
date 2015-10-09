#include <Data/Mesh/EdgeMesh.h>

#include <algorithm>
#include <iostream>

#include <GL/oglCommon.h>

using namespace Data;
using namespace Data::Mesh;

Edge::Edge( int _v0, int _v1 ) : v0(_v0), v1(_v1) { }

int Edge::GetSmaller() const { return (v0 < v1) ? v0 : v1; }
int Edge::GetLarger()  const { return (v0 < v1) ? v1 : v0; }

bool Edge::operator < ( const Edge & right ) const {
    if( GetSmaller() < right.GetSmaller() ) return true;
    if( GetSmaller() > right.GetSmaller() ) return false;
    if( GetLarger()  < right.GetLarger()  ) return true;
    return false;
}

bool Edge::operator == ( const Edge & right ) const {
    return ( GetSmaller() == right.GetSmaller() ) && ( GetLarger() == right.GetLarger() );
}

EdgeMesh::EdgeMesh( ){ }

Edge & EdgeMesh::GetElement( int id ){
    return at(id);
}

int EdgeMesh::GetElementCount( ){
    return (int)size();
}

void EdgeMesh::RemoveDuplicates( ){
    std::sort( begin(), end() );
    assign( begin(), std::unique( begin(), end() ) );
}

void EdgeMesh::Draw(const std::vector<int> &render_order, PointMesh &pmesh, ColorMesh &colors ){
    glColor4ubv( (GLubyte*) &colors.DefaultColor() );
    glBegin(GL_LINES);
    for(int i = 0; i < (int)render_order.size(); i++){
        int idx = render_order[i];
        Data::Mesh::Edge e = GetElement(idx);
        if( colors.isColorPerElement() ) glColor4ubv( (GLubyte*)&colors.GetColor( idx ) );
        if( colors.isColorPerVertex() )  glColor4ubv( (GLubyte*)&colors.GetColor( e.v0 ) );
        glVertex3fv( pmesh.GetVertex(e.v0).data );
        if( colors.isColorPerVertex() )  glColor4ubv( (GLubyte*)&colors.GetColor( e.v1 ) );
        glVertex3fv( pmesh.GetVertex(e.v1).data );
    }
    glEnd();

}


void EdgeMesh::Draw( PointMesh & pmesh, SCI::Vex4 color ){
    glColor4fv( color.data );
    glBegin(GL_LINES);
    for(int i = 0; i < size(); i++){
        Data::Mesh::Edge e = GetElement(i);
        glVertex3fv( pmesh.GetVertex(e.v0).data );
        glVertex3fv( pmesh.GetVertex(e.v1).data );
    }
    glEnd();
}

void EdgeMesh::SortByPainters(std::vector<int> &render_order, SCI::Mat4 proj, const PointMesh &pnts){
    std::vector<float> dpt;
    for(int i = 0; i < (int)pnts.points.size(); i++){
        dpt.push_back( (proj * pnts.points.at(i)).z );
    }

    std::vector< std::pair<float, int> > dpt_list;
    for( int t = 0; t < (int)size(); t++){
        float cdpt = 0;
        cdpt += dpt.at( at(t).v0 );
        cdpt += dpt.at( at(t).v1 );
        dpt_list.push_back( std::make_pair( cdpt, t ) );
    }

    std::sort( dpt_list.begin(), dpt_list.end() );

    render_order.resize( dpt_list.size() );
    for(int i = 0; i < (int)dpt_list.size(); i++){
        render_order[i] = dpt_list[dpt_list.size()-i-1].second;
    }
}

/*
void EdgeMesh::SortByPainters( SCI::Mat4 proj, Data::Mesh::PointMesh & pnts, Data::Mesh::ColorMesh & cm  ){

    std::vector<float> dpt;
    for(int i = 0; i < (int)pnts.points.size(); i++){
        dpt.push_back( (proj * pnts.points.at(i)).z );
    }

    std::vector< std::pair<float, int> > dpt_list;

    for( int t = 0; t < (int)size(); t++){
        float cdpt = 0;
        cdpt += dpt.at( at(t).v0 );
        cdpt += dpt.at( at(t).v1 );
        dpt_list.push_back( std::make_pair( cdpt, t ) );
    }

    std::sort( dpt_list.begin(), dpt_list.end() );

    std::vector<Edge> new_edge;
    for(int i = (int)dpt_list.size()-1; i >= 0; i--){
        new_edge.push_back( at( dpt_list[i].second ) );
    }
    clear();
    assign( new_edge.begin(), new_edge.end() );

    if( cm.isColorPerElement() ){
        std::vector<unsigned int> new_color;

        for(int i = (int)dpt_list.size()-1; i >= 0; i--){
            new_edge.push_back( at( dpt_list[i].second ) );
            new_color.push_back( cm.at( dpt_list[i].second ) );
        }

        cm.clear();
        cm.SetColorPerElement();
        cm.assign( new_color.begin(), new_color.end() );
    }

}
*/
