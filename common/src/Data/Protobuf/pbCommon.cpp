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

#include <Data/Protobuf/pbCommon.h>

#include <stdio.h>

#ifdef WIN32
    #define POPEN  _popen
    #define PCLOSE _pclose
#else
    #define POPEN  popen
    #define PCLOSE pclose
#endif


namespace Data {

    namespace Protobuf {

        void WritePB(FILE * outfile, ::google::protobuf::Message & msg ){
                int siz = msg.ByteSize();
                char * tmp = new char[siz];
                msg.SerializeToArray(tmp,siz);
                fwrite( &siz, sizeof(int),   1, outfile );
                fwrite(  tmp,           1, siz, outfile );
                delete [] tmp;
        }

        void WritePB_Type( FILE * outfile, unsigned int pb_type ){
                fwrite( &pb_type, sizeof(int), 1, outfile );
        }

        unsigned int ReadPB_Type( FILE * infile ){
                unsigned int ret;
                if( fread(&ret, sizeof(int),1, infile ) != 1){
                    return 0xffffffff;
                }
                return ret;
        }

        bool ReadPB(FILE * infile, ::google::protobuf::Message & msg ){
                int siz;
                if( fread(&siz,sizeof(int),1, infile) != 1 ){
                    printf("WARNING (ReadPB): Trouble reading size\n");
                    return false;
                }
                char * tmp = new char[siz];
                if( fread( tmp,1,siz,infile) != siz ){
                    printf("WARNING (ReadPB): Trouble reading message\n");
                    delete [] tmp;
                    return false;
                }
                bool ret = msg.ParseFromArray( tmp, siz );
                delete [] tmp;
                return ret;
        }

        bool ReadUnknownPB( FILE * infile ){
            int siz;
            if( fread(&siz,sizeof(int),1, infile) != 1 ){
                printf("WARNING (ReadUnknownPB): Trouble reading size\n");
                return false;
            }
            char * tmp = new char[siz];
            if( fread( tmp,1,siz,infile) != siz ){
                printf("WARNING (ReadUnknownPB): Trouble reading message\n");
                delete [] tmp;
                return false;
            }
            delete [] tmp;
            return true;
        }

    }

}
