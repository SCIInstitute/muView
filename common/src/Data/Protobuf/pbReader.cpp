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

#include <Data/Protobuf/pbReader.h>

#include <stdio.h>

#ifdef WIN32
    #define POPEN  _popen
    #define PCLOSE _pclose
#else
    #define POPEN  popen
    #define PCLOSE pclose
#endif


using namespace Data::Protobuf;


pbReader::pbReader( const char * fname ){
    gzip = false;
    infile = 0;
    if( fname ){
        open(fname);
    }
}

pbReader::~pbReader( ){
    close();
}

bool pbReader::open( const char * fname ){
    gzip = ( strcmp(fname+strlen(fname)-3,".gz") == 0 );

    if(gzip){
        char cmd[1024];
        sprintf(cmd,"gunzip -c %s",fname);
        infile = POPEN(cmd,"r");
    }
    else{
        infile = fopen(fname,"rb");
    }
    return infile != 0;
}

bool pbReader::close( ){
    if(infile){
        if(gzip){
            PCLOSE(infile);
        }
        else{
            fclose(infile);
        }
        infile = 0;
        return true;
    }
    return false;
}


unsigned int pbReader::pop_type( ){
    return Data::Protobuf::ReadPB_Type( infile );
}

bool pbReader::pop_message( ::google::protobuf::Message & msg ){
    if(!Data::Protobuf::ReadPB( infile, msg )) return false;
    return !isEOF();
}

bool pbReader::pop_message(){
    Data::Protobuf::ReadUnknownPB( infile );
    return !isEOF();
}

bool pbReader::isEOF(){
    if(infile==0) return true;
    return feof(infile) != 0;
}

