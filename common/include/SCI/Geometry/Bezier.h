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

#ifndef SCI_BEZIER_H
#define SCI_BEZIER_H

#include <SCI/Vex2.h>
#include <SCI/Vex3.h>
#include <SCI/Vex4.h>

namespace SCI {
    template <class C> C LinearBezier( float t, const C & p0, const C & p1 ){
        return p0 * (1-t) + p1 * (t);
    }

    template <class C> C QuadraticBezier(float t, const C & p0, const C & p1, const C & p2){
        return p0 * (1-t) * (1-t) + p1 * (1-t) * (t) * 2 + p2 * (t) * (t);
    }

    template <class C> C CubicBezier(float t, const C & p0, const C & p1, const C & p2, const C & p3){
        float t0 = t;
        float t1 = 1-t;
        C ret;
        ret += p0 * t1 * t1 * t1;
        ret += p1 * t1 * t1 * t0 * 3.0f;
        ret += p2 * t1 * t0 * t0 * 3.0f;
        ret += p3 * t0 * t0 * t0;
        return ret;
    }


    inline float LinearBezier(float t, float p0, float p1){    return LinearBezier<float>(t,p0,p1); }
    inline Vex2  LinearBezier(float t, Vex2  p0, Vex2  p1){    return LinearBezier<Vex2>(t,p0,p1);  }
    inline Vex3  LinearBezier(float t, Vex3  p0, Vex3  p1){    return LinearBezier<Vex3>(t,p0,p1);  }
    inline Vex4  LinearBezier(float t, Vex4  p0, Vex4  p1){    return LinearBezier<Vex4>(t,p0,p1);  }


    inline float QuadraticBezier(float t, float p0, float p1, float p2){ return QuadraticBezier<float>(t,p0,p1,p2); }
    inline Vex2  QuadraticBezier(float t, Vex2  p0, Vex2  p1, Vex2  p2){ return QuadraticBezier<Vex2>(t,p0,p1,p2);  }
    inline Vex3  QuadraticBezier(float t, Vex3  p0, Vex3  p1, Vex3  p2){ return QuadraticBezier<Vex3>(t,p0,p1,p2);  }
    inline Vex4  QuadraticBezier(float t, Vex4  p0, Vex4  p1, Vex4  p2){ return QuadraticBezier<Vex4>(t,p0,p1,p2);  }

    inline float CubicBezier(float t, float p0, float p1, float p2, float p3){ return CubicBezier<float>(t,p0,p1,p2,p3); }
    inline Vex2  CubicBezier(float t, Vex2  p0, Vex2  p1, Vex2  p2, Vex2  p3){ return CubicBezier<Vex2>(t,p0,p1,p2,p3);  }
    inline Vex3  CubicBezier(float t, Vex3  p0, Vex3  p1, Vex3  p2, Vex3  p3){ return CubicBezier<Vex3>(t,p0,p1,p2,p3);  }
    inline Vex4  CubicBezier(float t, Vex4  p0, Vex4  p1, Vex4  p2, Vex4  p3){ return CubicBezier<Vex4>(t,p0,p1,p2,p3);  }
}


#endif // SCI_BEZIER_H
