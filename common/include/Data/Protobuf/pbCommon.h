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

#ifndef DATA_PROTOBUF_PBCOMMON_H
#define DATA_PROTOBUF_PBCOMMON_H


#include <google/protobuf/stubs/common.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>

#define MESSAGE_GPU_FILE_TABLE               0x0001
#define MESSAGE_GPU_LINE_NUMBERS             0x0002
#define MESSAGE_GPU_OP                       0x0003
#define MESSAGE_COHOMOLOGY_PARAMETERIZATION  0x0010
#define MESSAGE_FILETABLE                    0x0012
#define MESSAGE_COHOMOLOGY_MTR               0x0014
#define MESSAGE_CPU_TRACERECORD              0x0020
    #define MESSAGE_CPUTRACERECORD           0x0020
#define MESSAGE_CPU_FILETABLE                0x0022
    #define MESSAGE_CPUFILETABLE             0x0022
#define MESSAGE_CPU_MEMORY_ALLOCATED         0x0024
#define MESSAGE_CPU_MEMORY_DEALLOCATED       0x0026
#define MESSAGE_CPU_FUNCTION_RECORD          0x0028
#define MESSAGE_CACHE_CONFIGURATION_RECORD   0x0030
#define MESSAGE_CACHE_INSTRUCTION_RECORD     0x0032


namespace Data {

    namespace Protobuf {

        void WritePB(FILE * outfile, ::google::protobuf::Message & msg );
        void WritePB_Type( FILE * outfile, unsigned int pb_type );
        unsigned int ReadPB_Type( FILE * infile );
        bool ReadPB(FILE * infile, ::google::protobuf::Message & msg );
        bool ReadUnknownPB( FILE * infile );

    }

}


#endif // DATA_PROTOBUF_PBCOMMON_H
