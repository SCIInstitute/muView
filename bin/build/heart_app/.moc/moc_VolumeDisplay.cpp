/****************************************************************************
** Meta object code from reading C++ file 'VolumeDisplay.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../include/Drawable/VolumeDisplay.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'VolumeDisplay.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Drawable__VolumeDisplay_t {
    QByteArrayData data[3];
    char stringdata0[35];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Drawable__VolumeDisplay_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Drawable__VolumeDisplay_t qt_meta_stringdata_Drawable__VolumeDisplay = {
    {
QT_MOC_LITERAL(0, 0, 23), // "Drawable::VolumeDisplay"
QT_MOC_LITERAL(1, 24, 9), // "ResetView"
QT_MOC_LITERAL(2, 34, 0) // ""

    },
    "Drawable::VolumeDisplay\0ResetView\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Drawable__VolumeDisplay[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void Drawable::VolumeDisplay::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        VolumeDisplay *_t = static_cast<VolumeDisplay *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->ResetView(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject Drawable::VolumeDisplay::staticMetaObject = {
    { &DrawableObject::staticMetaObject, qt_meta_stringdata_Drawable__VolumeDisplay.data,
      qt_meta_data_Drawable__VolumeDisplay,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *Drawable::VolumeDisplay::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Drawable::VolumeDisplay::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_Drawable__VolumeDisplay.stringdata0))
        return static_cast<void*>(const_cast< VolumeDisplay*>(this));
    return DrawableObject::qt_metacast(_clname);
}

int Drawable::VolumeDisplay::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = DrawableObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
