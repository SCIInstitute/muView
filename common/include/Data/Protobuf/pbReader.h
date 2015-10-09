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


#ifndef DATA_PROTOBUF_PBREADER_H
#define DATA_PROTOBUF_PBREADER_H

#include <Data/Protobuf/pbCommon.h>

namespace Data {

    namespace Protobuf {

        class pbReader {
        public:
            pbReader( const char * fname = 0 );
            ~pbReader( );

            bool open( const char * fname );
            bool close( );

            unsigned int pop_type( );
            bool         pop_message( );
            bool         pop_message( ::google::protobuf::Message & msg );
            bool         isEOF();

        protected:
            bool gzip;
            FILE * infile;
        };

    }

}

#endif // DATA_PROTOBUF_PBREADER_H
