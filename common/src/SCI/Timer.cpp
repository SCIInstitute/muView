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

#include <SCI/Timer.h>

using namespace SCI;

#if defined(WIN32)

    Timer::Timer(void){
            QueryPerformanceFrequency(&perfFreq);
            offset = 0;
            paused = true;
    }

    void Timer::Start(){
        QueryPerformanceCounter(&start);
        paused = false;
    }

    double Timer::Tick(){
            if(paused){ return offset; }
            QueryPerformanceCounter(&stop);
            return offset + (double)(stop.QuadPart - start.QuadPart)/(double)(perfFreq.QuadPart);
    }

    void Timer::Pause(){
            offset = Tick();
            paused = true;
    }

    void Timer::Resume(float speed){
            QueryPerformanceCounter(&start);
            paused = false;
    }

    void Timer::Step(float amount){
            offset += amount;
    }

    bool Timer::Paused(){ return paused; }

#endif

#if defined(linux) || defined(__APPLE__)

    #ifdef __ANDROID__
        //#include <jni.h>
        //#include <android/log.h>

        //#include <GLES2/gl2.h>
        //#include <GLES2/gl2ext.h>

        //#include <stdio.h>
        //#include <stdlib.h>
        //#include <math.h>
    #endif

    #include <stdio.h>
    #include <sys/time.h>

    Timer::Timer() : paused(true), offset(0.0){}

    void Timer::Start(){
        start = getSeconds();
        offset = 0.0;
        paused = false;
    }

    double Timer::Tick(){
        if(paused){
            return offset;
        }
        stop = getSeconds();
        return offset + (stop - start);
    }

    void Timer::Pause(){
        offset = Tick();
        paused = true;
    }

    void Timer::Resume(float speed){
        start = getSeconds();
        paused = false;
    }

    void Timer::Step(float amount){
        offset += amount;
    }

    bool Timer::Paused(){
        return paused;
    }

    double Timer::getSeconds(){
      struct timeval tv;

      gettimeofday(&tv, NULL);
      return tv.tv_sec + 1e-6*tv.tv_usec;
    }
#endif
