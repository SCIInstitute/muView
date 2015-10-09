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

#ifndef SCI_TIMER_H
#define SCI_TIMER_H

#if defined(WIN32)
    #include <windows.h>
#endif

namespace SCI {

    ///////////////////////////////////////////////////////////////////
    /// \brief High performance timing
    ///
    /// This class provides high performance timing for single events
    ///////////////////////////////////////////////////////////////////
    class Timer {

    public:
        ///////////////////////////////////////////////////////////////
        /// \brief Default Constructor
        ///////////////////////////////////////////////////////////////
        Timer(void);

        ///////////////////////////////////////////////////////////////
        /// \brief Triggers the start of an event.
        ///////////////////////////////////////////////////////////////
        void Start();

        ///////////////////////////////////////////////////////////////
        /// \brief Triggers the end of an event.
        ///////////////////////////////////////////////////////////////
        double Tick();

        void Pause();

        void Resume(float speed);

        void Step(float amount);

        bool Paused();

    protected:
        bool paused;
        double offset;
        float speed;

#if defined(__linux__) || defined(__APPLE__)
        double getSeconds();
        double start, stop;
    #endif

    #if defined(WIN32)
        LARGE_INTEGER start, stop, perfFreq;
    #endif

    };

}

#endif //SCI_TIMER_H
