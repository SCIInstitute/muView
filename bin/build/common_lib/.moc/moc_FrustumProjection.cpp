/****************************************************************************
** Meta object code from reading C++ file 'FrustumProjection.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../Common/include/SCI/Camera/FrustumProjection.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'FrustumProjection.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_SCI__FrustumProjection_t {
    QByteArrayData data[9];
    char stringdata0[72];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SCI__FrustumProjection_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SCI__FrustumProjection_t qt_meta_stringdata_SCI__FrustumProjection = {
    {
QT_MOC_LITERAL(0, 0, 22), // "SCI::FrustumProjection"
QT_MOC_LITERAL(1, 23, 7), // "SetHFOV"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 3), // "fov"
QT_MOC_LITERAL(4, 36, 7), // "SetVFOV"
QT_MOC_LITERAL(5, 44, 9), // "SetHither"
QT_MOC_LITERAL(6, 54, 5), // "znear"
QT_MOC_LITERAL(7, 60, 6), // "SetYon"
QT_MOC_LITERAL(8, 67, 4) // "zfar"

    },
    "SCI::FrustumProjection\0SetHFOV\0\0fov\0"
    "SetVFOV\0SetHither\0znear\0SetYon\0zfar"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SCI__FrustumProjection[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   34,    2, 0x0a /* Public */,
       4,    1,   37,    2, 0x0a /* Public */,
       5,    1,   40,    2, 0x0a /* Public */,
       7,    1,   43,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    6,
    QMetaType::Void, QMetaType::Double,    8,

       0        // eod
};

void SCI::FrustumProjection::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        FrustumProjection *_t = static_cast<FrustumProjection *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->SetHFOV((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->SetVFOV((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->SetHither((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->SetYon((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject SCI::FrustumProjection::staticMetaObject = {
    { &Projection::staticMetaObject, qt_meta_stringdata_SCI__FrustumProjection.data,
      qt_meta_data_SCI__FrustumProjection,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *SCI::FrustumProjection::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SCI::FrustumProjection::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_SCI__FrustumProjection.stringdata0))
        return static_cast<void*>(const_cast< FrustumProjection*>(this));
    return Projection::qt_metacast(_clname);
}

int SCI::FrustumProjection::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Projection::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
