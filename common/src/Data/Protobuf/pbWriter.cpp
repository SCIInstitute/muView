/*
**  Common Library
**  Copyright (C) 2013  Paul Rosen
**
**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <Data/Protobuf/pbWriter.h>

#include <stdio.h>

#ifdef WIN32
    #define POPEN  _popen
    #define PCLOSE _pclose
#else
    #define POPEN  popen
    #define PCLOSE pclose
#endif


using namespace Data::Protobuf;



pbWriter::pbWriter( const char * fname ){
    outfile = 0;
    gzip = false;
    if( fname ){
        open(fname);
    }
}

pbWriter::~pbWriter( ){
    close();
}

bool pbWriter::ready( ){
    return (outfile!=0);
}

bool pbWriter::open( const char * fname ){
    close();
    gzip = ( strcmp(fname+strlen(fname)-3,".gz") == 0 );
    if(gzip){
        char cmd[1024];
        sprintf(cmd,"gzip > %s",fname);
        outfile = POPEN( cmd, "w" );
    }
    else{
        outfile = fopen(fname,"wb");
    }
    return (outfile!=0);
}

bool pbWriter::close( ){
    if(outfile){
        if(gzip){
            PCLOSE(outfile);
        }
        else{
            fclose(outfile);
        }
        outfile = 0;
        return true;
    }
    return false;
}

bool pbWriter::push(unsigned int msg_type, google::protobuf::Message &msg ){
    Data::Protobuf::WritePB_Type( outfile, msg_type );
    Data::Protobuf::WritePB( outfile, msg );
    return true;
}


