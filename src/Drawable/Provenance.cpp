#include <Drawable/Provenance.h>

using namespace Drawable;

Provenance::Provenance(){
    head     = 0;
    curr     = 0;
    data_in  = 0;
    selected = false;
}

void Provenance::Initialize( Data::ProxyData &data ){

    // Set the input data
    data_in = &data;

    // Create the head node
    SCI::Subset ss( data.GetVoxelCount() );
    Push( BBox(), ss, 0 );

}

ProvenanceNode * Provenance::GetCurrentNode(){
    return curr;
}

void Provenance::Push( BBox bbox ){

    SCI::Subset newsubset;
    SCI::Vex2 pos;
    for(int i = 0; i < curr->GetDR()->GetElementCount(); i++ ){
        curr->GetDR()->GetElementPoint( i, pos.data );
        if( bbox.isContained( pos ) ){
            newsubset.push_back( curr->GetDR()->GetElementVoxID(i) );
        }
    }
    if( newsubset.size() > 0 ){
        Push( bbox, newsubset, curr->GetMethod() );
    }

}

void Provenance::Push( int dr_method ){
    SCI::Subset ss = curr->GetElementList();
    Push( BBox(), ss, dr_method );
}

void Provenance::Push( QAction * dr_method ){
    SCI::Subset ss = curr->GetElementList();
    Push( BBox(), ss, dr_method->data().toInt() );
}


void Provenance::Push( BBox bbox, SCI::Subset newelems, int dr_method ){

    // create new child node, set as current
    if(head == 0){
        head = curr = new ProvenanceNode( data_in, newelems, dr_method );
    }
    else{
        curr = curr->AddChild( newelems, dr_method, bbox );
    }
    head->UpdatePointers();

    std::vector<int> cur_column(head->CountDepth(),0);
    CalculateNodeLocation( head, cur_column, head->CountLevelWidth( ), 1, head->CountDepth( ) );
    for(int i = 0; i < 3; i++){
        CalculateNodeLocationPass2( head );
        CalculateNodeLocationPass3( head );
    }
    for(int i = 0; i < 3; i++){
        CalculateNodeLocationPass3( head );
    }

}

void Provenance::Draw( ){
    head->UpdateDrawingSurface();
    head->UpdateBins();

    SetViewport();

    Clear();

    float aspect = (float)height/(float)width;

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(-1,1,-aspect,aspect,-1,1);

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    float boundary = 0.5f;
    if( curr->location.x < -boundary ){
        float newx = SCI::lerp(curr->location.x,-boundary,0.1f);
        head->TranslateLocation( newx-curr->location.x, 0 );
    }
    if( curr->location.x > boundary ){
        float newx = SCI::lerp(curr->location.x,boundary,0.1f);
        head->TranslateLocation( newx-curr->location.x, 0 );
    }

    if( curr->location.y < -boundary*aspect ){
        float newy = SCI::lerp(curr->location.y,-boundary*aspect,0.1f);
        head->TranslateLocation( 0, newy-curr->location.y );
    }
    if( curr->location.y > boundary*aspect ){
        float newy = SCI::lerp(curr->location.y,boundary*aspect,0.1f);
        head->TranslateLocation( 0, newy-curr->location.y );
    }

    DrawNode( head );

    DrawFrame();
}

void Provenance::CalculateNodeLocation( ProvenanceNode * node, std::vector<int> & cur_column, std::vector<int> level_widths, int curdepth, int maxdepth ){

    //float aspect = (float)height/(float)width;

    float lft_x = -(float)(level_widths[curdepth-1]+1)*0.5f/2.0f;
    float cur_x = lft_x + (float)(cur_column[curdepth-1]+1)*0.5f;
    float top_y = (float)(maxdepth)*0.55f/2.0f;
    float cur_y = top_y - (float)curdepth*0.5f;

    node->location.x = cur_x;
    node->location.y = cur_y;

    cur_column[curdepth-1]++;
    for(int i = 0; i < (int)node->children.size(); i++){
        CalculateNodeLocation( node->children[i], cur_column, level_widths, curdepth+1, maxdepth) ;
    }
}

void Provenance::CalculateNodeLocationPass2( ProvenanceNode * node ){

    if(node->parent){
        float x_pos = 0;
        float cnt = 0;
        if((int)node->children.size() > 0){
            for(int i = 0; i < (int)node->children.size(); i++){
                x_pos += node->children[i]->location.x;
            }
            x_pos /= (float)node->children.size();
            cnt+=1;
        }
        node->location.x = (2.0f * node->location.x + x_pos + node->parent->location.x)/(3.0f+cnt);
    }

    for(int i = 0; i < (int)node->children.size(); i++){
        CalculateNodeLocationPass2( node->children[i] ) ;
    }

}

void Provenance::CalculateNodeLocationPass3( ProvenanceNode * node ){

    float x_pos = node->location.x;
    if( node->neighbor_left && (node->location.x-node->neighbor_left->location.x) < 0.6 ){
        float push = (node->location.x-node->neighbor_left->location.x) - 0.6;
        x_pos -= push/2.0f;
    }
    if( node->neighbor_right && (node->neighbor_right->location.x-node->location.x) < 0.6 ){
        float push = (node->neighbor_right->location.x-node->location.x) - 0.6;
        x_pos += push/2.0f;
    }
    node->location.x = x_pos;

    for(int i = 0; i < (int)node->children.size(); i++){
        CalculateNodeLocationPass3( node->children[i] ) ;
    }

}


void Provenance::DrawNode( ProvenanceNode * node ){
    if(node == 0) return;

    glPushMatrix();
        glTranslatef( node->location.x, node->location.y, 0 );
        glScalef( 0.2f, 0.2f, 1.0f );
        glColor3f(1,1,1);
        node->GetDrawingSurface().GetColorTexture(0)->DrawTexture();
    glPopMatrix();

    glLineWidth( 2.0f);
    glColor3f(0,0,0);
    if( node == curr ) glColor3f(1,0,1);
    glBegin(GL_LINE_LOOP);
        glVertex3f( node->location.x-0.2f, node->location.y-0.2f, 0 );
        glVertex3f( node->location.x+0.2f, node->location.y-0.2f, 0 );
        glVertex3f( node->location.x+0.2f, node->location.y+0.2f, 0 );
        glVertex3f( node->location.x-0.2f, node->location.y+0.2f, 0 );
    glEnd();

    for(int i = 0; i < (int)node->children.size(); i++){
        glColor3f(0,0,0);
        glBegin(GL_LINES);
            glVertex3f( node->location.x,          node->location.y-0.2f,          0 );
            glVertex3f( node->children[i]->location.x, node->children[i]->location.y+0.2f, 0 );
        glEnd();
        DrawNode( node->children[i] );
    }
}


bool Provenance::MouseClick(int button, int state, int x, int y){

    if( isPointInFrame( x,y ) ){
        selected = true;

        if( state == DrawableObject::Down ){
            float aspect = (float)height/(float)width;
            float fx = (float)(x-u0) / (float)width  * 2.0f - 1.0f;
            float fy = (float)(y-v0) / (float)height * 2.0f - 1.0f;
            ProvenanceNode * newcurr = head->GetSelected( SCI::Vex2( fx, fy*aspect ) );
            if(newcurr){
                curr = newcurr;
                return true;
            }
        }
    }

    return false;
}

bool Provenance::Keypress( int key, int x, int y ){
    if( selected ){
        if(key == Key_Delete){
            printf("delete pressed\n"); fflush(stdout);
            return true;
        }
    }
    return false;
}

bool Provenance::ClearSelection( ){
    selected = false;
    return true;
}

