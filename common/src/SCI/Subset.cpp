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

#include <SCI/Subset.h>

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <algorithm>

using namespace SCI;

Subset::Subset( int elemN ) {
    for(int i = 0; i < elemN; i++){
        push_back(i);
    }
}

void Subset::GetRandomSubset( Subset & src, int elemN ){
    clear();
    std::map<int,bool> tmp;
    while( (int)size() < elemN && (int)size() < (int)src.size() ){
        int num = src[ abs(rand()*rand()+rand())%src.size() ];
        if( tmp.find(num) == tmp.end() ){
            tmp[num] = true;
            push_back( num );
        }
    }
}

void Subset::GetStridedSubset( Subset & src, int stride, int elemN ){
    clear();
    for(int i = 0; i < elemN; i++){
        push_back( src[ (i*stride)%src.size() ] );
    }
}

void Subset::GetStridedSubset( Subset & src, int stride ){
    clear();
    for(int i = 0; i < (int)src.size(); i+=stride){
        push_back( src[ i ] );
    }
}


void Subset::GetChunkSubset( Subset & src, int start, int elemN ){
    clear();
    for(int i = 0; i < elemN; i++){
        push_back( src[ (start+i)%src.size() ] );
    }
}

void Subset::Sort( ){
    std::sort( begin(), end() );
}

