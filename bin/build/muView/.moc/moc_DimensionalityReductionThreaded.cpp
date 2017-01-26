/****************************************************************************
** Meta object code from reading C++ file 'DimensionalityReductionThreaded.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../include/DimensionalityReductionThreaded.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'DimensionalityReductionThreaded.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_DimensionalityReductionThreaded_t {
    QByteArrayData data[4];
    char stringdata0[74];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DimensionalityReductionThreaded_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DimensionalityReductionThreaded_t qt_meta_stringdata_DimensionalityReductionThreaded = {
    {
QT_MOC_LITERAL(0, 0, 31), // "DimensionalityReductionThreaded"
QT_MOC_LITERAL(1, 32, 19), // "computationComplete"
QT_MOC_LITERAL(2, 52, 0), // ""
QT_MOC_LITERAL(3, 53, 20) // "prematureTermination"

    },
    "DimensionalityReductionThreaded\0"
    "computationComplete\0\0prematureTermination"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DimensionalityReductionThreaded[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,

       0        // eod
};

void DimensionalityReductionThreaded::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        DimensionalityReductionThreaded *_t = static_cast<DimensionalityReductionThreaded *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->computationComplete((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (DimensionalityReductionThreaded::*_t)(bool );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&DimensionalityReductionThreaded::computationComplete)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject DimensionalityReductionThreaded::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_DimensionalityReductionThreaded.data,
      qt_meta_data_DimensionalityReductionThreaded,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *DimensionalityReductionThreaded::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DimensionalityReductionThreaded::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_DimensionalityReductionThreaded.stringdata0))
        return static_cast<void*>(const_cast< DimensionalityReductionThreaded*>(this));
    return QThread::qt_metacast(_clname);
}

int DimensionalityReductionThreaded::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
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

// SIGNAL 0
void DimensionalityReductionThreaded::computationComplete(bool _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
