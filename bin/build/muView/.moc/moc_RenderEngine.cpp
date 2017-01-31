/****************************************************************************
** Meta object code from reading C++ file 'RenderEngine.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../include/muView/RenderEngine.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RenderEngine.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_RenderEngine_t {
    QByteArrayData data[45];
    char stringdata0[763];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RenderEngine_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RenderEngine_t qt_meta_stringdata_RenderEngine = {
    {
QT_MOC_LITERAL(0, 0, 12), // "RenderEngine"
QT_MOC_LITERAL(1, 13, 17), // "setDrawModePoints"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 26), // "setDrawModeVolumeRendering"
QT_MOC_LITERAL(4, 59, 23), // "setDrawModeIsosurfacing"
QT_MOC_LITERAL(5, 83, 24), // "setDrawModeDistanceField"
QT_MOC_LITERAL(6, 108, 18), // "setDrawModeNetwork"
QT_MOC_LITERAL(7, 127, 21), // "setColorModeDimension"
QT_MOC_LITERAL(8, 149, 15), // "setColorModeMin"
QT_MOC_LITERAL(9, 165, 18), // "setColorModeMedian"
QT_MOC_LITERAL(10, 184, 15), // "setColorModeMax"
QT_MOC_LITERAL(11, 200, 17), // "setColorModeStDev"
QT_MOC_LITERAL(12, 218, 19), // "setColorModeCluster"
QT_MOC_LITERAL(13, 238, 20), // "setColorModeIsovalue"
QT_MOC_LITERAL(14, 259, 15), // "setColorModePCA"
QT_MOC_LITERAL(15, 275, 18), // "setColorModeFibers"
QT_MOC_LITERAL(16, 294, 12), // "setDimension"
QT_MOC_LITERAL(17, 307, 15), // "setClusterCount"
QT_MOC_LITERAL(18, 323, 20), // "setClusterIterations"
QT_MOC_LITERAL(19, 344, 21), // "setClusterRecalculate"
QT_MOC_LITERAL(20, 366, 19), // "setClusterHistogram"
QT_MOC_LITERAL(21, 386, 20), // "setClusterTypeL2Norm"
QT_MOC_LITERAL(22, 407, 21), // "setClusterTypePearson"
QT_MOC_LITERAL(23, 429, 23), // "setClusterTypeHistogram"
QT_MOC_LITERAL(24, 453, 17), // "setFiberDirection"
QT_MOC_LITERAL(25, 471, 14), // "setFiberLength"
QT_MOC_LITERAL(26, 486, 11), // "setIsovalue"
QT_MOC_LITERAL(27, 498, 16), // "setDimIsosurface"
QT_MOC_LITERAL(28, 515, 16), // "setMinIsosurface"
QT_MOC_LITERAL(29, 532, 17), // "setMeanIsosurface"
QT_MOC_LITERAL(30, 550, 16), // "setMaxIsosurface"
QT_MOC_LITERAL(31, 567, 11), // "setClipXVal"
QT_MOC_LITERAL(32, 579, 11), // "setClipYVal"
QT_MOC_LITERAL(33, 591, 11), // "setClipZVal"
QT_MOC_LITERAL(34, 603, 12), // "setClipXFlip"
QT_MOC_LITERAL(35, 616, 12), // "setClipYFlip"
QT_MOC_LITERAL(36, 629, 12), // "setClipZFlip"
QT_MOC_LITERAL(37, 642, 14), // "setClipXEnable"
QT_MOC_LITERAL(38, 657, 14), // "setClipYEnable"
QT_MOC_LITERAL(39, 672, 14), // "setClipZEnable"
QT_MOC_LITERAL(40, 687, 18), // "calculateSubVolume"
QT_MOC_LITERAL(41, 706, 25), // "UpdateRenderEngine2DColor"
QT_MOC_LITERAL(42, 732, 15), // "RenderEngine2D*"
QT_MOC_LITERAL(43, 748, 2), // "re"
QT_MOC_LITERAL(44, 751, 11) // "Recalculate"

    },
    "RenderEngine\0setDrawModePoints\0\0"
    "setDrawModeVolumeRendering\0"
    "setDrawModeIsosurfacing\0"
    "setDrawModeDistanceField\0setDrawModeNetwork\0"
    "setColorModeDimension\0setColorModeMin\0"
    "setColorModeMedian\0setColorModeMax\0"
    "setColorModeStDev\0setColorModeCluster\0"
    "setColorModeIsovalue\0setColorModePCA\0"
    "setColorModeFibers\0setDimension\0"
    "setClusterCount\0setClusterIterations\0"
    "setClusterRecalculate\0setClusterHistogram\0"
    "setClusterTypeL2Norm\0setClusterTypePearson\0"
    "setClusterTypeHistogram\0setFiberDirection\0"
    "setFiberLength\0setIsovalue\0setDimIsosurface\0"
    "setMinIsosurface\0setMeanIsosurface\0"
    "setMaxIsosurface\0setClipXVal\0setClipYVal\0"
    "setClipZVal\0setClipXFlip\0setClipYFlip\0"
    "setClipZFlip\0setClipXEnable\0setClipYEnable\0"
    "setClipZEnable\0calculateSubVolume\0"
    "UpdateRenderEngine2DColor\0RenderEngine2D*\0"
    "re\0Recalculate"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RenderEngine[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      41,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,  219,    2, 0x0a /* Public */,
       3,    0,  220,    2, 0x0a /* Public */,
       4,    0,  221,    2, 0x0a /* Public */,
       5,    0,  222,    2, 0x0a /* Public */,
       6,    0,  223,    2, 0x0a /* Public */,
       7,    0,  224,    2, 0x0a /* Public */,
       8,    0,  225,    2, 0x0a /* Public */,
       9,    0,  226,    2, 0x0a /* Public */,
      10,    0,  227,    2, 0x0a /* Public */,
      11,    0,  228,    2, 0x0a /* Public */,
      12,    0,  229,    2, 0x0a /* Public */,
      13,    0,  230,    2, 0x0a /* Public */,
      14,    0,  231,    2, 0x0a /* Public */,
      15,    0,  232,    2, 0x0a /* Public */,
      16,    1,  233,    2, 0x0a /* Public */,
      17,    1,  236,    2, 0x0a /* Public */,
      18,    1,  239,    2, 0x0a /* Public */,
      19,    0,  242,    2, 0x0a /* Public */,
      20,    1,  243,    2, 0x0a /* Public */,
      21,    0,  246,    2, 0x0a /* Public */,
      22,    0,  247,    2, 0x0a /* Public */,
      23,    0,  248,    2, 0x0a /* Public */,
      24,    1,  249,    2, 0x0a /* Public */,
      25,    1,  252,    2, 0x0a /* Public */,
      26,    1,  255,    2, 0x0a /* Public */,
      27,    1,  258,    2, 0x0a /* Public */,
      28,    1,  261,    2, 0x0a /* Public */,
      29,    1,  264,    2, 0x0a /* Public */,
      30,    1,  267,    2, 0x0a /* Public */,
      31,    1,  270,    2, 0x0a /* Public */,
      32,    1,  273,    2, 0x0a /* Public */,
      33,    1,  276,    2, 0x0a /* Public */,
      34,    0,  279,    2, 0x0a /* Public */,
      35,    0,  280,    2, 0x0a /* Public */,
      36,    0,  281,    2, 0x0a /* Public */,
      37,    1,  282,    2, 0x0a /* Public */,
      38,    1,  285,    2, 0x0a /* Public */,
      39,    1,  288,    2, 0x0a /* Public */,
      40,    0,  291,    2, 0x0a /* Public */,
      41,    1,  292,    2, 0x0a /* Public */,
      44,    0,  295,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 42,   43,
    QMetaType::Void,

       0        // eod
};

void RenderEngine::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        RenderEngine *_t = static_cast<RenderEngine *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->setDrawModePoints(); break;
        case 1: _t->setDrawModeVolumeRendering(); break;
        case 2: _t->setDrawModeIsosurfacing(); break;
        case 3: _t->setDrawModeDistanceField(); break;
        case 4: _t->setDrawModeNetwork(); break;
        case 5: _t->setColorModeDimension(); break;
        case 6: _t->setColorModeMin(); break;
        case 7: _t->setColorModeMedian(); break;
        case 8: _t->setColorModeMax(); break;
        case 9: _t->setColorModeStDev(); break;
        case 10: _t->setColorModeCluster(); break;
        case 11: _t->setColorModeIsovalue(); break;
        case 12: _t->setColorModePCA(); break;
        case 13: _t->setColorModeFibers(); break;
        case 14: _t->setDimension((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 15: _t->setClusterCount((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: _t->setClusterIterations((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: _t->setClusterRecalculate(); break;
        case 18: _t->setClusterHistogram((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: _t->setClusterTypeL2Norm(); break;
        case 20: _t->setClusterTypePearson(); break;
        case 21: _t->setClusterTypeHistogram(); break;
        case 22: _t->setFiberDirection((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 23: _t->setFiberLength((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 24: _t->setIsovalue((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 25: _t->setDimIsosurface((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 26: _t->setMinIsosurface((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 27: _t->setMeanIsosurface((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 28: _t->setMaxIsosurface((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 29: _t->setClipXVal((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 30: _t->setClipYVal((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 31: _t->setClipZVal((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 32: _t->setClipXFlip(); break;
        case 33: _t->setClipYFlip(); break;
        case 34: _t->setClipZFlip(); break;
        case 35: _t->setClipXEnable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 36: _t->setClipYEnable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 37: _t->setClipZEnable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 38: _t->calculateSubVolume(); break;
        case 39: _t->UpdateRenderEngine2DColor((*reinterpret_cast< RenderEngine2D*(*)>(_a[1]))); break;
        case 40: _t->Recalculate(); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 39:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< RenderEngine2D* >(); break;
            }
            break;
        }
    }
}

const QMetaObject RenderEngine::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_RenderEngine.data,
      qt_meta_data_RenderEngine,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *RenderEngine::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RenderEngine::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_RenderEngine.stringdata0))
        return static_cast<void*>(const_cast< RenderEngine*>(this));
    return QObject::qt_metacast(_clname);
}

int RenderEngine::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 41)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 41;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 41)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 41;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
