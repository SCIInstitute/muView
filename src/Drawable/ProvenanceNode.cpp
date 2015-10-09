#include <Drawable/ProvenanceNode.h>

#include <Data/MappedData.h>
#include <GL/oglCommon.h>

using namespace Drawable;

//ProvenanceNode::ProvenanceNode( Data::BasicData * _data_in, Subset & _elements, int method ) : histogram(_data_in->GetDim(),256,256) {
ProvenanceNode::ProvenanceNode( Data::ProxyData * _data_in, SCI::Subset & _elements, int method ) : histogram(_data_in->GetDim(),64,64) {
    data_out = 0;
    dr = 0;
    fbo_init = false;
    curdraw = 0;
    parent         = 0;
    neighbor_left  = 0;
    neighbor_right = 0;


    data_in = _data_in;

    // Set the output data
    data_out = new Data::MappedData( data_in->GetX(), data_in->GetY(), data_in->GetZ(), 2 );

    // Create a set containing all voxels
    elements = _elements;

    // Create a subset with random elements from the complete set
    if(method == 0){
        features.GetRandomSubset( elements, STD_MIN( (int)elements.size(), 200 ) );
    }
    else{
        features.GetRandomSubset( elements, STD_MIN( (int)elements.size(), 200 ) );
    }

    // Run PCA as our default DR approach
    dr = new RemoteDimensionalityReduction( );
    QObject::connect( dr, SIGNAL(computationComplete(bool)), this, SLOT(ComputationCompleted(bool)) );
    dr->Start( method, *data_in, features, *data_out, elements );
    bbmin.Set( dr->GetMinValue(0), dr->GetMinValue(1) );
    bbmax.Set( dr->GetMaxValue(0), dr->GetMaxValue(1) );
    curdraw = 0;

    curbin    = 0;


    if( !fbo_init ){
        fbo.Init( 128, 128 );
        fbo.InsertColorAttachment( 0 );
        fbo.InsertDepthAttachment( );
        fbo.GetColorTexture(0)->SetMinMagFilter( GL_LINEAR, GL_LINEAR );
        //fbo.GetColorTexture(0)->SetMinMagFilter( GL_NEAREST, GL_NEAREST );
        fbo_init = true;

        fbo.Enable( true );

        glClearColor(1,1,1,1);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        fbo.Disable( );
    }


}

void ProvenanceNode::ComputationCompleted(bool prematureTermination){
    printf("COMPUTATION COMPLETED %s\n",prematureTermination?"EARLY":"");
    fflush(stdout);
}

void ProvenanceNode::UpdatePointers(){
    int depth = CountDepth();
    std::vector<int> cur_column(depth,1);
    std::map< std::pair<int,int>, ProvenanceNode* > map;
    BuildMap( map, cur_column, 1 );
    cur_column = std::vector<int>(depth,1);
    UpdatePointers( map, 0, cur_column, 1 );
}

void ProvenanceNode::UpdatePointers( std::map< std::pair<int,int>, ProvenanceNode* > map, ProvenanceNode* _parent, std::vector<int> & cur_column, int curdepth ){
    if( map.find( std::pair<int,int>(curdepth,cur_column[curdepth-1]-1) ) != map.end() ){
        neighbor_left = map[ std::pair<int,int>(curdepth,cur_column[curdepth-1]-1) ];
    }
    if( map.find( std::pair<int,int>(curdepth,cur_column[curdepth-1]+1) ) != map.end() ){
        neighbor_right = map[ std::pair<int,int>(curdepth,cur_column[curdepth-1]+1) ];
    }
    parent = _parent;
    cur_column[curdepth-1]++;
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->UpdatePointers( map, this, cur_column, curdepth+1 );
    }
}

void ProvenanceNode::BuildMap( std::map< std::pair<int,int>, ProvenanceNode* > & map, std::vector<int> & cur_column, int curdepth ){
    map[ std::pair<int,int>(curdepth,cur_column[curdepth-1]) ] = this;
    cur_column[curdepth-1]++;
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->BuildMap( map, cur_column, curdepth+1 );
    }
}

int ProvenanceNode::GetMethod() const {
    return dr->GetCurrentMethod();
}

const RemoteDimensionalityReduction *ProvenanceNode::GetDR() const{
    return dr;
}
int ProvenanceNode::CountDepth( ) const {
    int maxdepth = 1;
    for(int i = 0; i < (int)children.size(); i++){
        maxdepth = STD_MAX( maxdepth, 1 + children[i]->CountDepth() );
    }
    return maxdepth;
}

void ProvenanceNode::TranslateLocation( float dx, float dy ){
    location.x += dx;
    location.y += dy;
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->TranslateLocation(dx,dy);
    }
}

void ProvenanceNode::CountLevelWidth( int curdepth, std::vector<int> & list ) const {
    list[curdepth-1]++;
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->CountLevelWidth( curdepth+1, list );
    }
}

std::vector<int> ProvenanceNode::CountLevelWidth( ) const {
    std::vector<int> ret( CountDepth( ), 0 );
    CountLevelWidth( 1, ret );
    return ret;
}

SCI::Mat4 ProvenanceNode::GetTransform() const {
    return transform;
}

SCI::Subset ProvenanceNode::GetFeatureList() const {
    return features;
}

SCI::Subset ProvenanceNode::GetElementList() const {
    return elements;
}


ProvenanceNode * ProvenanceNode::AddChild( SCI::Subset & _elements, int method, BBox bbox ){
    children.push_back( new ProvenanceNode( data_in, _elements, method ) );
    child_bboxes.push_back( bbox );
    curdraw = 0;
    curbin  = 0;
    return children.back();
}

void ProvenanceNode::UpdateDrawingSurface( ){
    GetDrawingSurface();

    for(int i = 0; i < (int)children.size(); i++){
        children[i]->UpdateDrawingSurface();
    }

    if( curdraw != 0 && curdraw == dr->GetElementCount() ){
        return;
    }

    float ox = -(bbmax.x+bbmin.x)/2.0f;
    float oy = -(bbmax.y+bbmin.y)/2.0f;
    float sx = 1.8f / (bbmax.x-bbmin.x);
    float sy = 1.8f / (bbmax.y-bbmin.y);

    transform.LoadIdentity();
    transform *= SCI::Mat4( SCI::Mat4::MAT4_SCALE,     sx, sy, 1.0f );
    transform *= SCI::Mat4( SCI::Mat4::MAT4_TRANSLATE, ox, oy, 1.0f );

    fbo.Enable( true );

    if(curdraw == 0){
        glClearColor(1,1,1,1);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }

    glEnable(GL_DEPTH_TEST);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glPointSize(2.0f);
    glPushMatrix();

        glMultMatrixf( transform.data );

        glLineWidth( 1.0f );
        for(int i = 0; i < (int)child_bboxes.size(); i++){
            glColor3f(0.8f,0.8f,1.0f);
            glBegin(GL_LINE_LOOP);
                glVertex3f( child_bboxes[i].bbmin.x, child_bboxes[i].bbmin.y, -0.2f );
                glVertex3f( child_bboxes[i].bbmax.x, child_bboxes[i].bbmin.y, -0.2f );
                glVertex3f( child_bboxes[i].bbmax.x, child_bboxes[i].bbmax.y, -0.2f );
                glVertex3f( child_bboxes[i].bbmin.x, child_bboxes[i].bbmax.y, -0.2f );
            glEnd();
            glColor3f(0.9f,0.9f,1.0f);
            glBegin(GL_QUADS);
                glVertex3f( child_bboxes[i].bbmin.x, child_bboxes[i].bbmin.y, -0.1f );
                glVertex3f( child_bboxes[i].bbmax.x, child_bboxes[i].bbmin.y, -0.1f );
                glVertex3f( child_bboxes[i].bbmax.x, child_bboxes[i].bbmax.y, -0.1f );
                glVertex3f( child_bboxes[i].bbmin.x, child_bboxes[i].bbmax.y, -0.1f );
            glEnd();
        }

        glBegin(GL_POINTS);

        SCI::Vex3 pnt;

        glColor3f(0,0,0);
        if(curdraw == 0){
            for(int i = 0; i < dr->GetFeatureCount(); i++){
                dr->GetFeaturePoint( i, pnt.data );
                glVertex3f( pnt.x, pnt.y, -0.3f );
            }
        }

        glColor3f(1,0,1);
        for(; curdraw < dr->GetElementCount(); curdraw++){
            dr->GetElementPoint( curdraw, pnt.data );
            glVertex3f( pnt.x, pnt.y,  -0.5f );
        }

        glEnd();

    glPopMatrix();

    glLineWidth(6.0f);
    float pct = (float)(dr->GetElementCount()+dr->GetFeatureCount())/(float)(dr->GetTotalElementCount()+dr->GetFeatureCount());
    glColor3f(0.7f,0.7f,0.7f);
    glBegin(GL_LINES);
        glVertex3f(-1,-1,0);
        glVertex3f(-1+2.0f*pct,-1,0);
    glEnd();

    glDisable(GL_DEPTH_TEST);

    fbo.Disable();


    SCI::Vex2 tbbmin( dr->GetMinValue(0), dr->GetMinValue(1) );
    SCI::Vex2 tbbmax( dr->GetMaxValue(0), dr->GetMaxValue(1) );

    if( (tbbmin-bbmin).Length() > 0.001f || (tbbmax-bbmax).Length() > 0.001f ){
        bbmin = tbbmin;
        bbmax = tbbmax;
        curdraw = 0;
        curbin  = 0;
    }

}

void ProvenanceNode::UpdateBins(){
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->UpdateBins();
    }

    int dim = data_in->GetDim();

    if( curbin == 0 ){
        histogram.Clear();
    }

    SCI::Vex2 pnt;
    float * space = new float[dim];
    for(; curbin < dr->GetElementCount(); curbin++){
        dr->GetElementPoint( curbin, pnt.data );
        pnt = (transform * SCI::Vex3(pnt,0)).xy();
        data_in->GetElement( dr->GetElementVoxID( curbin ), space );
        histogram.AddElement(pnt,space);
    }

}

bool ProvenanceNode::isDone() const {
    return (dr->GetElementCount() == dr->GetTotalElementCount());
}

oglWidgets::oglFrameBufferObject & ProvenanceNode::GetDrawingSurface( ) {
    if( !fbo_init ){
        fbo.Init( 128, 128 );
        fbo.InsertColorAttachment( 0 );
        fbo.InsertDepthAttachment( );
        fbo.GetColorTexture(0)->SetMinMagFilter( GL_LINEAR, GL_LINEAR );
        //fbo.GetColorTexture(0)->SetMinMagFilter( GL_NEAREST, GL_NEAREST );
        fbo_init = true;

        fbo.Enable( true );

        glClearColor(1,1,1,1);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        fbo.Disable( );
    }
    return fbo;
}


const ProvenanceNode * ProvenanceNode::GetSelected( SCI::Vex2 loc ) const {
    if( fabsf(location.x-loc.x) <= 0.2f && fabsf(location.y-loc.y) <= 0.2f ){
        return this;
    }
    for(int i = 0; i < (int)children.size(); i++){
        const ProvenanceNode * tmp = children[i]->GetSelected( loc );
        if(tmp) return tmp;
    }
    return 0;
}


ProvenanceNode * ProvenanceNode::GetSelected( SCI::Vex2 loc ) {
    if( fabsf(location.x-loc.x) <= 0.2f && fabsf(location.y-loc.y) <= 0.2f ){
        return this;
    }
    for(int i = 0; i < (int)children.size(); i++){
        ProvenanceNode * tmp = children[i]->GetSelected( loc );
        if(tmp) return tmp;
    }
    return 0;
}

void ProvenanceNode::ColorData( unsigned int * col ){
    for(int i = 0; i < (int)elements.size(); i++){
        SCI::Vex3 pos;
        data_out->GetElement( elements[i], pos.data );
        pos = transform * pos;
        col[ elements[i] ] = line.GetColor( pos.xy() ).UIntColor();
    }
    for(int i = 0; i < (int)children.size(); i++){
        children[i]->ColorData(col);
    }
}
