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

#ifndef SUBSET_H
#define SUBSET_H

#include <vector>

namespace SCI {
    class Subset : public std::vector<int> {

    public:
        // Creates a subset with elements numbered Zero to elemN-1
        Subset( int elemN = 0 );

        // Gets a random subset of data from src
        void GetRandomSubset(  Subset & src, int elemN );

        // Gets a strided subset of data from src
        void GetStridedSubset( Subset & src, int stride, int elemN );

        // Gets a strided subset of data from src
        void GetStridedSubset( Subset & src, int stride );

        // Gets a contiguous chunk of data from src
        void GetChunkSubset(   Subset & src, int start, int elemN );

        void Sort( );
    };
}

#endif // SUBSET_H
