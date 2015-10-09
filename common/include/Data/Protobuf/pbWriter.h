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


#ifndef DATA_PROTOBUF_PBWRITER_H
#define DATA_PROTOBUF_PBWRITER_H

#include <Data/Protobuf/pbCommon.h>

namespace Data {
    namespace Protobuf {

        class pbWriter {
        public:
            pbWriter( const char * fname = 0 );
            ~pbWriter( );

            bool open( const char * fname );
            bool close( );

            bool ready( );

            bool push( unsigned int msg_type, ::google::protobuf::Message & msg );

        protected:
            bool gzip;
            FILE * outfile;
        };

    }

}

#endif // DATA_PROTOBUF_PBWRITER_H
