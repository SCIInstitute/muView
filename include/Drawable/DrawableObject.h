#ifndef DRAWABLEOBJECT_H
#define DRAWABLEOBJECT_H

#include <QObject>

namespace Drawable {

    // Objects which implement some kind of basic drawing ability
    class DrawableObject : public QObject {

        Q_OBJECT

    public:
        const static int NoButton;
        const static int LeftButton;
        const static int RightButton;
        const static int MiddleButton;
        const static int XButton1;
        const static int XButton2;

        const static int Down;
        const static int Up;
        const static int Double;

        const static int Key_Delete;
        const static int Key_Enter;
        const static int Key_Backspace;


    public:
        DrawableObject();

        virtual void Draw( ) = 0;

        virtual bool MouseClick(int button, int state, int x, int y);
        virtual bool MouseMotion(int button, int x, int dx, int y, int dy);

        virtual bool Keypress( int key, int x, int y );
        virtual bool ClearSelection( );

        virtual void SetShape( int _u0, int _v0, int w, int h );

        bool isPointInFrame( int x, int y );

    protected:
        int u0, v0, width, height;

        void SetViewport();
        void Clear( );
        void DrawFrame();

    };

}

#endif // DRAWABLEOBJECT_H
