#pragma once

#include <SCI/Vex2.h>
#include <SCI/Vex3.h>
#include <SCI/Vex4.h>
#include <SCI/Mat4.h>

#define ZERO_TOLERANCE 1.0e-10

///////////////////////////////////////////////////////////////////
/// \brief Calculate the distance between a line and point in 2D
///
/// Calculates the distance between a line (not segment) and a 
/// point in 2D
///
/// \param v0 Any point on the line
/// \param v1 A different point on the line
/// \param p  A point anywhere in 2D space
///
/// \return The distance of the point to the line
///////////////////////////////////////////////////////////////////
inline float PointLineDistance(const SCI::Vex2 & v0, const SCI::Vex2 & v1, const SCI::Vex2 & p){
    SCI::Vex2 v1v0 = v1-v0;
    SCI::Vex2 pv0  = p-v0;

    float u = dot(pv0,v1v0) / dot(v1v0,v1v0);
    SCI::Vex2 p_close = lerp(v0,v1,u);

    return (p_close - p).Length();
}

inline float PointLineDistance(const SCI::Vex3 & v0, const SCI::Vex3 & v1, const SCI::Vex3 & p){
    SCI::Vex3 v1v0 = v1-v0;
    SCI::Vex3 pv0  = p-v0;

    float u = dot(pv0,v1v0) / dot(v1v0,v1v0);
    SCI::Vex3 p_close = lerp(v0,v1,u);

    return (p_close - p).Length();
}


inline int QuadraticEquation(float a, float b, float c, float & t0, float & t1){
    float tmp = (b*b) - 4.0f*a*c;
    t0 = t1 = FLT_MAX;

    if(fabsf(a) < ZERO_TOLERANCE){ return 0; }

    if(fabsf(tmp) < ZERO_TOLERANCE){
        t0 = -b / (2.0f * a);
        return 1;
    }

    if(tmp < 0){ return 0; }

    t0 = (-b + sqrtf(tmp)) / (2.0f * a);
    t1 = (-b - sqrtf(tmp)) / (2.0f * a);

    return 2;
}

///////////////////////////////////////////////////////////////////
/// \brief Calculate the distance between a segment and point in 2D
///
/// Calculates the distance between a segment and a point in 2D
///
/// \param v0 Start point of the segment
/// \param v1 End point of the segment
/// \param p  A point anywhere in 2D space
///
/// \return The distance of the point to the segment
///////////////////////////////////////////////////////////////////
inline float PointSegmentDistance(const SCI::Vex2 & v0, const SCI::Vex2 & v1, const SCI::Vex2 & p){
    SCI::Vex2 v1v0 = v1-v0;
    SCI::Vex2 v0p  = v0-p;
    SCI::Vex2 v1p  = v1-p;

    float u = -dot(v0p,v1v0) / dot(v1v0,v1v0);

    if(u >= 1){ return v1p.Length(); }
    if(u <= 0){ return v0p.Length(); }

    SCI::Vex2 p_close = lerp(v0,v1,u);

    return (p_close - p).Length();
}

inline float PointSegmentDistance(const SCI::Vex3 & v0, const SCI::Vex3 & v1, const SCI::Vex3 & p){
    SCI::Vex3 v1v0 = v1-v0;
    SCI::Vex3 v0p  = v0-p;
    SCI::Vex3 v1p  = v1-p;

    float u = -dot(v0p,v1v0) / dot(v1v0,v1v0);

    if(u >= 1){ return v1p.Length(); }
    if(u <= 0){ return v0p.Length(); }

    SCI::Vex3 p_close = lerp(v0,v1,u);

    return (p_close - p).Length();
}


///////////////////////////////////////////////////////////////////
/// \brief Calculate the distance between 2 2D points
///
/// \param v0 First point in space
/// \param v1 Second point in space
///
/// \return Distance between the 2 points
///////////////////////////////////////////////////////////////////
inline float PointPointDistance(const SCI::Vex2 & v0, const SCI::Vex2 & v1){
    return (v1-v0).Length();
}

///////////////////////////////////////////////////////////////////
/// \brief Calculate the distance between 2 3D points
///
/// \param v0 First point in space
/// \param v1 Second point in space
///
/// \return Distance between the 2 points
///////////////////////////////////////////////////////////////////
inline float PointPointDistance(const SCI::Vex3 & v0, const SCI::Vex3 & v1){
    return (v1-v0).Length();
}

///////////////////////////////////////////////////////////////////
/// \brief Calculate the area of the 3D triangle
///
/// \param v0 Vertex 0
/// \param v1 Vertex 1
/// \param v2 Vertex 2
///
/// \return area of the triangle
///////////////////////////////////////////////////////////////////
inline float TriangleArea(const SCI::Vex3& v0, const SCI::Vex3& v1, const SCI::Vex3& v2){
    float a = (v1-v0).Length();
    float b = (v2-v1).Length();
    float c = (v0-v2).Length();

    float s = 0.5f * (a + b + c);

    return sqrtf(s*(s-a)*(s-b)*(s-c));
}

inline float TriangleArea(const SCI::Vex2 & v0, const SCI::Vex2 & v1, const SCI::Vex2 & v2){
    float a = (v1-v0).Length();
    float b = (v2-v1).Length();
    float c = (v0-v2).Length();

    float s = 0.5f * (a + b + c);

    return sqrtf(s*(s-a)*(s-b)*(s-c));
}


///////////////////////////////////////////////////////////////////
/// \brief Calculate the plane formed by a 3D point and a normal
///
/// Calculates the constants (ABCD) of the plane equation 
/// Ax + By + Cz + D = 0
///
/// \param p0 Point on the plane
/// \param norm Normal of the plane
///
/// \return 4D vector with the values of ABCD
///////////////////////////////////////////////////////////////////
inline SCI::Vex4 CalculatePlane(const SCI::Vex3 & p0, const SCI::Vex3 & norm){
    return SCI::Vex4(norm,-dot(norm,p0));
}

///////////////////////////////////////////////////////////////////
/// \brief Calculate the plane formed by 3-3D points
///
/// Calculates the constants (ABCD) of the plane equation 
/// Ax + By + Cz + D = 0
/// Vertices are expected in counter-clockwise order.
///
/// \param p0 First point in plane
/// \param p1 Second point in plane
/// \param p2 Third point in plane
///
/// \return 4D vector with the values of ABCD
///////////////////////////////////////////////////////////////////
inline SCI::Vex4 CalculatePlane(const SCI::Vex3 & p0, const SCI::Vex3 & p1, const SCI::Vex3 & p2){
    SCI::Vex3 p1p0 = p1 - p0;
    SCI::Vex3 p2p0 = p2 - p0;
    SCI::Vex3 norm = normalize(cross(p1p0,p2p0));

    return CalculatePlane(p0,norm);
}


///////////////////////////////////////////////////////////////////
/// \brief Calculate the intersection of a ray and plane in 3D
///
/// \param r0 Start point of the ray
/// \param rd Direction (and magnitude) of the ray (e.g. r1 - r0)
/// \param plane Plane formed by the plane equation 
///        Ax + By + Cz + D = 0
///
/// \return The interpolation value t of the intersection.  The 
///         intersection equals r0 + t * rd.  With values of t < 0 
///         or t > 1, the plane and ray do not intersect.  When the 
///         ray and plane are near parallel, -1 is returned.
///////////////////////////////////////////////////////////////////
inline float RayPlaneIntersection(const SCI::Vex3 & r0, const SCI::Vex3 & rd, const SCI::Vex4 & plane){
    float ndotr = dot(plane.xyz(),rd);

    if(fabsf(ndotr) < ZERO_TOLERANCE){ return FLT_MAX; }

    return -dot(plane,SCI::Vex4(r0,1)) / ndotr;
}


inline float PointPlaneDistance(const SCI::Vex3 & p, const SCI::Vex3 & p0, const SCI::Vex3 & norm){
    return dot(norm,p - p0);
}

inline float PointPlaneDistance(const SCI::Vex3 & p, const SCI::Vex4 & pln){
    return dot(pln,SCI::Vex4(p,1));
}

inline SCI::Vex3 MirrorPointAcrossPlane(const SCI::Vex3 & p, const SCI::Vex4 & pln){
    float d = RayPlaneIntersection(p, -pln.xyz(), pln );

    return p - pln.xyz() * (d * 2.0f);

/*
    float ndotr = dot(plane.xyz(),rd);

    if(fabsf(ndotr) < ZERO_TOLERANCE){ return FLT_MAX; }

    return -dot(plane,SCI::Vex4(r0,1)) / ndotr;
*/
}


inline float LineLineIntersection(const SCI::Vex2 & p0, const SCI::Vex2 & pd, const SCI::Vex2 & r0, const SCI::Vex2 & rd){
    float de = cross(pd,rd);

    if(abs(de) < ZERO_TOLERANCE){ return FLT_MAX; }

    return cross(rd,p0-r0) / de;
}

inline SCI::Vex2 LineLineIntersectionPoint(const SCI::Vex2 & p0, const SCI::Vex2 & pd, const SCI::Vex2 & r0, const SCI::Vex2 & rd){
    float de = cross(pd,rd);

    if(abs(de) < ZERO_TOLERANCE){ return SCI::VEX2_INVALID; }

    return p0 + pd * cross(rd,p0-r0) / de;
}

inline float SegmentLineIntersection(const SCI::Vex2 & s0, const SCI::Vex2 & s1, const SCI::Vex2 & l0, const SCI::Vex2 & ld){
    float de = cross(s1-s0,ld);

    if(abs(de) < ZERO_TOLERANCE){ return FLT_MAX; }

    float ua = cross(ld,s0-l0) / de;

    if(ua < 0 || ua > 1){ return FLT_MAX; }

    return ua;
}

inline float SegmentSegmentIntersection(const SCI::Vex2 & p0, const SCI::Vex2 & p1, const SCI::Vex2 & r0, const SCI::Vex2 & r1){
    SCI::Vex2 r1r0 = r1 - r0;
    SCI::Vex2 p1p0 = p1 - p0;
    SCI::Vex2 p0r0 = p0 - r0;

    float de = cross(p1p0,r1r0);

    if(abs(de) < ZERO_TOLERANCE){ return FLT_MAX; }

    float ua = cross(r1r0,p0r0) / de;
    if((ua+ZERO_TOLERANCE) < 0 || (ua-ZERO_TOLERANCE) > 1){ return FLT_MAX; }

    float ub = cross(p1p0,p0r0) / de;
    if((ub+ZERO_TOLERANCE) < 0 || (ub-ZERO_TOLERANCE) > 1){ return FLT_MAX; }

    return SCI::Clamp(ua,0.0f,1.0f);
}

inline SCI::Vex2 SegmentSegmentIntersectionPoint(const SCI::Vex2 & p0, const SCI::Vex2 & p1, const SCI::Vex2 & r0, const SCI::Vex2 & r1){
    float t = SegmentSegmentIntersection(p0,p1,r0,r1);
    if(t >= 0 && t <= 1.0f){
        return lerp(p0,p1,t);
    }
    return SCI::Vex2(FLT_MAX,FLT_MAX);
}

inline float SegmentSegmentIntersection(const SCI::Vex2 p[2], const SCI::Vex2 r[2]){
    return SegmentSegmentIntersection(p[0],p[1],r[0],r[1]);
}

inline SCI::Vex2 SegmentSegmentIntersectionPoint(const SCI::Vex2 p[2], const SCI::Vex2 r[2]){
    float t = SegmentSegmentIntersection(p[0],p[1],r[0],r[1]);
    if(t >= 0 && t <= 1.0f){
        return lerp(p[0],p[1],t);
    }
    return SCI::Vex2(FLT_MAX,FLT_MAX);
}

inline void PlanePlaneIntersection(const SCI::Vex4 & p1, const SCI::Vex4 & p2, SCI::Vex3 & p, SCI::Vex3 & u){
    u = cross(p1.xyz(),p2.xyz());

    SCI::Vex3 p1w = p1.xyz() * p2.w;
    SCI::Vex3 p2w = p2.xyz() * p1.w;
    SCI::Vex3 p12w = p1w - p2w;

    if(fabsf(u.x) > fabsf(u.y) && fabsf(u.x) > fabsf(u.z)){
        p.x =  0;
        p.y =  p12w.z / u.x;
        p.z = -p12w.y / u.x;
    }
    else if(fabsf(u.y) > fabsf(u.z)){
        p.x = -p12w.z / u.y;
        p.y =  0;
        p.z =  p12w.x / u.y;
    }
    else{
        p.x =  p12w.y / u.z;
        p.y = -p12w.x / u.z;
        p.z =  0;
    }

    u.Normalize();
}

inline SCI::Vex3 PlanePlanePlaneIntersection(const SCI::Vex4 & p0, const SCI::Vex4 & p1, const SCI::Vex4 & p2){
    SCI::Vex3 p,u;
    PlanePlaneIntersection(p0,p1,p,u);
    return p + u * RayPlaneIntersection(p,u,p2);
}


// http://local.wasp.uwa.edu.au/~pbourke/geometry/2circle/
inline int CircleCircleIntersection(const SCI::Vex2 & c0, const float r0, const SCI::Vex2 & c1, const float r1, SCI::Vex2 & p0, SCI::Vex2 & p1){
    float d = (c0-c1).Length();

    // too far away
    if(d > (r0+r1)){ return 0; }
    // one circle contains the other
    if(d < fabsf(r0-r1)){ return 0; }

    // circles touch from outside
    if(d == (r0+r1)){
        p0 = p1 = c0 + (c1 - c0) * r0 / d;
        return 1;
    }
    // circles touch from inside
    if(d == (r0-r1)){
        p0 = p1 = c0 + (c1 - c0) * r0 / d;
        return 1;
    }

    float a = (r0*r0 - r1*r1 + d*d) / (2*d);
    float h = sqrtf(r0*r0 - a*a);

    SCI::Vex2 c2 = c0 + (c1 - c0) * a / d;
    SCI::Vex2 t = ( c1 - c0 ) * (h / d);

    p0.x = c2.x + t.y;
    p0.y = c2.y - t.x;

    p1.x = c2.x - t.y;
    p1.y = c2.y + t.x;

    return 2;
}

// http://mathforum.org/library/drmath/view/55103.html
inline int CircleTangentPoints(const SCI::Vex2 & c, const float r, const SCI::Vex2 & p, SCI::Vex2 & p0, SCI::Vex2 & p1){
    float r1 = (p-c).Length()/2.0f;
    SCI::Vex2 c1 = (p+c)/2.0f;

    return CircleCircleIntersection(c,r,c1,r1,p0,p1);
}



// http://local.wasp.uwa.edu.au/~pbourke/geometry/sphereline/
inline int CircleLineIntersection(const SCI::Vex2 & cen, const float r, const SCI::Vex2 & r0, const SCI::Vex2 & rd, float & t0, float & t1){
    float a = dot( rd, rd );
    float b = 2.0f * dot( rd, r0-cen );
    float c = dot( cen, cen ) + dot( r0, r0 ) - 2.0f * dot( cen, r0 ) - r*r;
    return  QuadraticEquation(a,b,c,t0,t1);
}

inline int CircleLineIntersection(const float r, const SCI::Vex2 & r0, const SCI::Vex2 & rd, float & t0, float & t1){
    float a = dot( rd, rd );
    float b = 2.0f * dot( rd, r0 );
    float c = dot( r0, r0 ) - r*r;
    return  QuadraticEquation(a,b,c,t0,t1);
}

inline int CircleLineIntersection(const float r, const SCI::Vex2 & r0, const SCI::Vex2 & rd, SCI::Vex2 & ret0, SCI::Vex2 & ret1){
    float t0 = 0, t1 = 0;
    int intersections = CircleLineIntersection(r,r0,rd,t0,t1);

    if(intersections >= 1){ ret0 = r0 + rd * t0; }
    if(intersections >= 2){ ret1 = r0 + rd * t1; }

    return intersections;
}

inline int CircleSegmentIntersection(const SCI::Vex2 & cen, const float r, const SCI::Vex2 & r0, const SCI::Vex2 & r1, float & t0, float & t1){
    float tt[2] = {FLT_MAX, FLT_MAX};
    int intersections = CircleLineIntersection(cen,r,r0,r1-r0,t0,t1);
    int ret = 0;

    if(intersections >= 1 && t0 >= 0 && t0 <= 1.0f){ tt[ret++] = t0; }
    if(intersections >= 2 && t1 >= 0 && t1 <= 1.0f){ tt[ret++] = t1; }

    t0 = tt[0];
    t1 = tt[1];

    return ret;
}

inline int CircleSegmentIntersection(const SCI::Vex2 & cen, const float r, const SCI::Vex2 & r0, const SCI::Vex2 & r1, SCI::Vex2 & ret0, SCI::Vex2 & ret1){
    float t0,t1;
    int intersections = CircleSegmentIntersection(cen,r,r0,r1,t0,t1);

    if(intersections >= 1){ ret0 = lerp(r0,r1,t0); }
    if(intersections >= 2){ ret1 = lerp(r0,r1,t1); }

    return intersections;
}


inline int CircleSegmentIntersection(const float r, const SCI::Vex2 & r0, const SCI::Vex2 & r1, float & t0, float & t1){
    float tt[2] = {FLT_MAX, FLT_MAX};
    int intersections = CircleLineIntersection(r,r0,r1-r0,t0,t1);
    int ret = 0;

    if(intersections >= 1 && t0 >= 0 && t0 <= 1.0f){ tt[ret++] = t0; }
    if(intersections >= 2 && t1 >= 0 && t1 <= 1.0f){ tt[ret++] = t1; }

    t0 = tt[0];
    t1 = tt[1];

    return ret;
}

inline int CircleSegmentIntersection(const float r, const SCI::Vex2 & r0, const SCI::Vex2 & r1, SCI::Vex2 & ret0, SCI::Vex2 & ret1){
    float t0,t1;
    int intersections = CircleSegmentIntersection(r,r0,r1,t0,t1);

    if(intersections >= 1){ ret0 = lerp(r0,r1,t0); }
    if(intersections >= 2){ ret1 = lerp(r0,r1,t1); }

    return intersections;
}




inline void BarycentricParameters(const SCI::Vex3& v1, const SCI::Vex3& v2, const SCI::Vex3& v3, SCI::Vex4& param0, SCI::Vex4& param1){

    SCI::Vex3 ADG = v3 - v1;
    SCI::Vex3 BEH = v2 - v3;

    float d0 = dot( ADG.yz(), BEH.xx() ) - dot( ADG.xx(), BEH.yz() );
    float d1 = dot( ADG.xz(), BEH.yy() ) - dot( ADG.yy(), BEH.xz() );

    SCI::Vex3 _r0,_r1;

    if(fabsf(d0) > fabsf(d1)){
        _r0 = SCI::Vex3( BEH.y+BEH.z,-BEH.x,-BEH.x) / d0;
        _r1 = SCI::Vex3( ADG.y+ADG.z,-ADG.x,-ADG.x) / d0;
    }
    else{
        _r0 = SCI::Vex3(-BEH.y, BEH.x+BEH.z,-BEH.y) / d1;
        _r1 = SCI::Vex3(-ADG.y, ADG.x+ADG.z,-ADG.y) / d1;
    }

    param0 = SCI::Vex4(_r0,-dot(_r0,v3));
    param1 = SCI::Vex4(_r1,-dot(_r1,v3));

}

inline SCI::Vex3 BarycentricCoordinates(const SCI::Vex3& p, const SCI::Vex4& param0, const SCI::Vex4& param1){
    SCI::Vex3 ret;

    ret.x = dot(param0,SCI::Vex4(p,1));
    ret.y = dot(param1,SCI::Vex4(p,1));
    ret.z = 1 - ret.y - ret.x;

    return ret;
}

inline SCI::Vex3 BarycentricCoordinates(const SCI::Vex3& p, const SCI::Vex3& v1, const SCI::Vex3& v2, const SCI::Vex3& v3){

    SCI::Vex4 param0,param1;

    BarycentricParameters(v1,v2,v3,param0,param1);

    return BarycentricCoordinates(p,param0,param1);
}


inline SCI::Vex3 TriangleSegmentIntersection(SCI::Vex3 r0, SCI::Vex3 r1, SCI::Vex4 plane, SCI::Vex3 p0, SCI::Vex3 p1, SCI::Vex3 p2){
    float t = RayPlaneIntersection(r0,r1-r0,plane);
    if(t < 0 || t > 1){
        return SCI::Vex3(-1,-1,-1);
    }
    SCI::Vex3 p = r0 + (r1-r0)*t;
    return BarycentricCoordinates(p,p0,p1,p2);
}

inline bool SegmentSphereIntersection(SCI::Vex3 r0, SCI::Vex3 r1, SCI::Vex3 cen, float rad){
    SCI::Vex3  rd  = r1 - r0;
    SCI::Vex3  r0c = r0 - cen;
    SCI::Vex3  r1c = r1 - cen;

    if(r0c.Length() < rad) return true;
    if(r1c.Length() < rad) return true;

    float A = dot(rd,rd);
    float B = dot(rd,r0c) * 2;
    float C = dot(r0c,r0c) - (rad * rad);

    float discriminant = B*B - 4*A*C;

    if(discriminant >= 0){
        float t0 = (-B + sqrtf(discriminant)) / (2*A);
        float t1 = (-B - sqrtf(discriminant)) / (2*A);
        if((t0 >= 0 && t0 <= 1.0f) || (t1 >= 0 && t1 <= 1.0f)) return true;
    }

    return false;
}


inline bool RayTriangleIntersection(const SCI::Vex3& r0, const SCI::Vex3& r1, const SCI::Vex3& v0, const SCI::Vex3& v1, const SCI::Vex3& v2, float& t, SCI::Vex3& b){
    SCI::Vex4 plane = CalculatePlane(v0,v1,v2);
    t = RayPlaneIntersection(r0,r1-r0,plane);
    SCI::Vex3 p = lerp(r0,r1,t);
    b = BarycentricCoordinates(p,v0,v1,v2);
    //Mat3 bm = BarycentricMatrix(v0,v1,v2);
    //bm.Print();
    //b = BarycentricCoordinates(bm,p);
    return (t >= 0 && t <= 1) && (b.x >= 0 && b.y >= 0 && b.z >= 0);
}


//#include <mylib/vector.h>

inline int PolygonCollapse(std::vector<SCI::Vex3> & pIn, std::vector<SCI::Vex3> & pOut){

    int size = (int)pIn.size();
    pOut.clear();
    int ret = 0;

    for(int i = 0; i < size; i++){
        int pa = (i-1+size)%size;
        int pb = (i+1)%size;

        SCI::Vex3 d0 = (pIn[i] -pIn[pa]).UnitVector();
        SCI::Vex3 d1 = (pIn[pb]-pIn[pa]).UnitVector();

        if(dot(d0,d1)<=0.9999f){
            pOut.push_back( pIn[i] );
        }
        else{
            ret++;
        }
    }
    return ret;
}

inline void PolygonCollapse(std::vector<SCI::Vex3> & pInOut){
    int loop = 0;
    std::vector<SCI::Vex3> ptmp;
    do {
        loop = PolygonCollapse(pInOut,ptmp);
        pInOut = ptmp;
    } while(loop > 0);
}

inline void PolygonClip(std::vector<SCI::Vex3> & pIn, std::vector<SCI::Vex3> & pOut, SCI::Vex4 plane){
    std::vector<float> side;

    for(int i = 0; i < pIn.size(); i++){
        side.push_back( SCI::dot(SCI::Vex4(pIn[i],1),plane) );
    }

    pOut.clear();
    for(int i = 0; i < pIn.size(); i++){
        int j = (i+1)%pIn.size();
        if(side[i] >= 0){ pOut.push_back( pIn[i] ); }
        if(side[i] >= 0 && side[j] <  0){ pOut.push_back( SCI::lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])) ); }
        if(side[i] <  0 && side[j] >= 0){ pOut.push_back( SCI::lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])) ); }
    }
}

inline void PolygonClip(std::vector<SCI::Vex3> & pInOut, SCI::Vex4 plane){
    std::vector<SCI::Vex3> ptmp;
    PolygonClip(pInOut,ptmp,plane);
    pInOut = ptmp;
}


inline void ClipFrustumWithPlane(std::vector<SCI::Vex3> & pIn, std::vector<SCI::Vex3> & pOut, const SCI::Vex3 cop, const SCI::Vex4 plane){
    float side[64];
    for(int i = 0; i < (int)pIn.size(); i++){
        side[i] = dot(SCI::Vex4(pIn[i],1),plane);
    }

    pOut.clear();
    for(int i = 0; i < pIn.size(); i++){
        int j = (i+1)%pIn.size();
        if(side[i] >= 0){ pOut.push_back( pIn[i] ); }
        if(side[i] <  0){
            float t = RayPlaneIntersection(cop,(pIn[i]-cop),plane);
            t = (t<0||t>10)?10:t;
            pOut.push_back( cop + (pIn[i]-cop) * t );
        }
        if(side[i] >= 0 && side[j] <  0){ pOut.push_back( SCI::lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])) ); }
        if(side[i] <  0 && side[j] >= 0){ pOut.push_back( SCI::lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])) ); }
    }
}


inline void ClipFrustumWithPlane(std::vector<SCI::Vex3> & pInOut, const SCI::Vex3 cop, const SCI::Vex4 plane){
    std::vector<SCI::Vex3> ptmp;

    ClipFrustumWithPlane(pInOut,ptmp,cop,plane);

    pInOut = ptmp;
}

inline int PointsOnPlaneCount(std::vector<SCI::Vex3> & pIn, const SCI::Vex4 plane, float eps){
    int ret = 0;
    for(int i = 0; i < pIn.size(); i++){
        float side = SCI::dot(SCI::Vex4(pIn[i],1),plane);
        if(fabsf(side) <= eps){
            ret++;
        }
    }
    return ret;
}

inline void PointsOnPlane(std::vector<SCI::Vex3> & pIn, std::vector<SCI::Vex3> & pOut, const SCI::Vex4 plane, float eps){
    pOut.clear();
    for(int i = 0; i < pIn.size(); i++){
        float side = SCI::dot(SCI::Vex4(pIn[i],1),plane);
        if(fabsf(side) <= eps){
            pOut.push_back(pIn[i]);
        }
    }
}

inline void PointsOnPlane(std::vector<SCI::Vex3> & pInOut, const SCI::Vex4 plane, float eps){
    std::vector<SCI::Vex3> ptmp;
    PointsOnPlane(pInOut,ptmp,plane,eps);
    pInOut = ptmp;
}


//
//inline void PolygonClip(SCI::Vex3* pIn, int pInN, SCI::Vex3* pOut, int & pOutN, SCI::Vex4 plane){
//    float side[64];
//    for(int i = 0; i < pInN; i++){
//        side[i] = dot(SCI::Vex4(pIn[i],1),plane);
//    }
//
//    pOutN = 0;
//    for(int i = 0; i < pInN; i++){
//        int j = (i+1)%pInN;
//        if(side[i] >= 0){ pOut[pOutN++] = pIn[i]; }
//        if(side[i] >= 0 && side[j] <  0){ pOut[pOutN++] = lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])); }
//        if(side[i] <  0 && side[j] >= 0){ pOut[pOutN++] = lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])); }
//    }
//}
//
//inline void PolygonClip(SCI::Vex3* p, int & pN, SCI::Vex4 plane){
//    SCI::Vex3 ptmp[64];
//    int ptmpN = 0;
//    PolygonClip(p,pN,ptmp,ptmpN,plane);
//
//    COPY_ARRAY(ptmp,ptmpN,p);
//    pN = ptmpN;
//}


//inline void ClipFrustumWithPlane(const SCI::Vex3 * pIn, const int pInN, SCI::Vex3 * pOut, int & pOutN, const SCI::Vex3 cop, const SCI::Vex4 plane){
//    float side[64];
//    for(int i = 0; i < pInN; i++){
//        side[i] = dot(SCI::Vex4(pIn[i],1),plane);
//    }
//
//    pOutN = 0;
//    for(int i = 0; i < pInN; i++){
//        int j = (i+1)%pInN;
//        if(side[i] >= 0){ pOut[pOutN++] = pIn[i]; }
//        if(side[i] <  0){
//            float t = RayPlaneIntersection(cop,(pIn[i]-cop),plane);
//            t = (t<0||t>10)?10:t;
//            pOut[pOutN++] = cop + (pIn[i]-cop) * t;
//        }
//        if(side[i] >= 0 && side[j] <  0){ pOut[pOutN++] = lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])); }
//        if(side[i] <  0 && side[j] >= 0){ pOut[pOutN++] = lerp(pIn[i],pIn[j],(0-side[i])/(side[j]-side[i])); }
//    }
//}

//inline void ClipFrustumWithPlane(SCI::Vex3 * pInOut, int & pInOutN, const SCI::Vex3 cop, const SCI::Vex4 plane){
//    SCI::Vex3 ptmp[64];
//    int ptmpN;
//
//    ClipFrustumWithPlane(pInOut,pInOutN,ptmp,ptmpN,cop,plane);
//
//    pInOutN = ptmpN;
//    COPY_ARRAY(ptmp,ptmpN,pInOut);
//}

//inline void PointsOnPlane(const SCI::Vex3 * pIn, const int pInN, SCI::Vex3 * pOut, int & pOutN, const SCI::Vex4 plane, float eps){
//    pOutN = 0;
//    for(int i = 0; i < pInN; i++){
//        float side = dot(SCI::Vex4(pIn[i],1),plane);
//        if(fabsf(side) <= eps){
//            pOut[pOutN++] = pIn[i];
//        }
//    }
//}
//
//inline void PointsOnPlane(SCI::Vex3 * pInOut, int & pInOutN, const SCI::Vex4 plane, float eps){
//    SCI::Vex3 ptmp[64];
//    int ptmpN;
//
//    PointsOnPlane(pInOut,pInOutN,ptmp,ptmpN,plane,eps);
//
//    pInOutN = ptmpN;
//    COPY_ARRAY(ptmp,ptmpN,pInOut);
//}

inline SCI::Vex3 ClosestPointBetweenLines(const SCI::Vex3 & P0, const SCI::Vex3 & u, const SCI::Vex3 & Q0, const SCI::Vex3 & v){
    SCI::Vex3 w0 = P0 - Q0;

    float a = dot(u,u);
    float b = dot(u,v);
    float c = dot(v,v);
    float d = dot(u,w0);
    float e = dot(v,w0);

    float s = (b*e - c*d) / (a*c - b*b);
    float t = (a*e - b*d) / (a*c - b*b);

    SCI::Vex3 p0 = P0 + u*s;
    SCI::Vex3 p1 = Q0 + v*t;

    return (p0+p1)/2.0f;
}



inline void ClipLine(SCI::Vex3 & p0, SCI::Vex3 & p1, SCI::Vex4 pln){
    float d0 = dot(SCI::Vex4(p0,1),pln);
    float d1 = dot(SCI::Vex4(p1,1),pln);
    if(d0 >= 0 && d1 >= 0){
        return;
    }
    if(d0 <= 0 && d1 <= 0){
        p0 = p1 = SCI::VEX3_INVALID;
        return;
    }
    if(d0 <= 0 && d1 >= 0){
        p0 = p0 + (p1-p0) * RayPlaneIntersection(p0,p1-p0,pln);
        return;
    }
    if(d0 >= 0 && d1 <= 0){
        p1 = p0 + (p1-p0) * RayPlaneIntersection(p0,p1-p0,pln);
        return;
    }
}





inline int __RayConeIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 c_origin, SCI::Vex3 c_axis, float cosAngle, float & t0, float & t1){
    float fCosSqr = cosAngle*cosAngle;
    SCI::Vex3 kE = r0 - c_origin;

    float fAdD = dot(c_axis,rd.UnitVector());
    float fEdA = dot(kE,c_axis);
    float fEdD = dot(kE,rd.UnitVector());
    float fEdE = dot(kE,kE);

    float A = 0.5f*(fAdD*fAdD - fCosSqr) * rd.Length();
    float B = 1.0f*(fAdD*fEdA - fCosSqr*fEdD);
    float C = 0.5f*(fEdA*fEdA - fCosSqr*fEdE) / rd.Length();

    int intersections = 0;

    if (fabsf(A) >= ZERO_TOLERANCE)
    {
        intersections = QuadraticEquation(A,B,C,t0,t1);
    }
    else if (fabsf(B) >= ZERO_TOLERANCE)
    {
        t0 = (C/B);
        intersections = 1;
    }

    float ret[2] = { FLT_MAX, FLT_MAX };
    int ret_cnt = 0;

    //if(intersections >= 1){
    //    SCI::Vex3 pnt = r0 + rd * t0;
    //    float diff = dot((pnt-c_origin).UnitVector(),c_axis)-cosAngle;
    //    if(fabsf(diff) < ZERO_TOLERANCE){ ret[ret_cnt++] = t0; }
    //}
    //if(intersections >= 2){
    //    SCI::Vex3 pnt = r0 + rd * t1;
    //    float diff = dot((pnt-c_origin).UnitVector(),c_axis)-cosAngle;
    //    if(fabsf(diff) < ZERO_TOLERANCE){ ret[ret_cnt++] = t1; }
    //}

    if(intersections >= 1){
        SCI::Vex3 pnt = r0 + rd * t0;
        float dir = dot(pnt-c_origin,c_axis)*cosAngle;
        if(dir >= 0){ ret[ret_cnt++] = t0; }
    }
    if(intersections >= 2){
        SCI::Vex3 pnt = r0 + rd * t1;
        float dir = dot(pnt-c_origin,c_axis)*cosAngle;
        if(dir >= 0){ ret[ret_cnt++] = t1; }
    }


    t0 = ret[0];
    t1 = ret[1];

    return ret_cnt;
}

inline int RayConeIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 c_origin, SCI::Vex3 c_axis, float c_angle, float & t0, float & t1){
    float cosAngle = cosf(c_angle);

    return __RayConeIntersection(r0,rd,c_origin,c_axis,cosAngle,t0,t1);
}

inline int RayConeIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 c_origin, SCI::Vex3 c_axis, SCI::Vex3 c_pnt, float & t0, float & t1){

    float cosAngle = dot((c_pnt-c_origin).UnitVector(),c_axis.UnitVector());

    return __RayConeIntersection(r0,rd,c_origin,c_axis,cosAngle,t0,t1);
}





//
//
//inline int __DoubleConeLineIntersection(SCI::Vex3 c_origin, SCI::Vex3 c_axis, float cosAngle, SCI::Vex3 r0, SCI::Vex3 rd, float & t0, float & t1){
//    float fCosSqr = cosAngle*cosAngle;
//    SCI::Vex3 kE = r0 - c_origin;
//
//    float fAdD = dot(c_axis,rd.UnitVector());
//    float fEdA = dot(kE,c_axis);
//    float fEdD = dot(kE,rd.UnitVector());
//    float fEdE = dot(kE,kE);
//
//    float A = 0.5f*(fAdD*fAdD - fCosSqr) * rd.Length();
//    float B = 1.0f*(fAdD*fEdA - fCosSqr*fEdD);
//    float C = 0.5f*(fEdA*fEdA - fCosSqr*fEdE) / rd.Length();
//
//    if (fabsf(A) >= ZERO_TOLERANCE)
//    {
//        return QuadraticEquation(A,B,C,t0,t1);
//    }
//    else if (fabsf(B) >= ZERO_TOLERANCE)
//    {
//        t0 = (C/B);
//        return 1;
//    }
//    return 0;
//}
//
//inline int __ConeLineIntersection(SCI::Vex3 c_origin, SCI::Vex3 c_axis, float cosAngle, SCI::Vex3 r0, SCI::Vex3 rd, float & t0, float & t1){
//
//    int intersections = __DoubleConeLineIntersection(c_origin, c_axis, cosAngle, r0, rd, t0, t1);
//
//    float ret[2] = { FLT_MAX, FLT_MAX };
//    int ret_cnt = 0;
//
//    if(intersections >= 1){
//        SCI::Vex3 pnt = r0 + rd * t0;
//        float diff = dot((pnt-c_origin).UnitVector(),c_axis)-cosAngle;
//        if(fabsf(diff) < ZERO_TOLERANCE){ ret[ret_cnt++] = t0; }
//    }
//    if(intersections >= 2){
//        SCI::Vex3 pnt = r0 + rd * t1;
//        float diff = dot((pnt-c_origin).UnitVector(),c_axis)-cosAngle;
//        if(fabsf(diff) < ZERO_TOLERANCE){ ret[ret_cnt++] = t1; }
//    }
//
//    t0 = ret[0];
//    t1 = ret[1];
//
//    return ret_cnt;
//}
//
//inline int __ConeSegmentIntersection(SCI::Vex3 c_origin, SCI::Vex3 c_axis, float cosAngle, SCI::Vex3 r0, SCI::Vex3 r1, float & t0, float & t1){
//
//    int intersections = __ConeLineIntersection(c_origin, c_axis, cosAngle, r0, r1-r0, t0, t1);
//
//    float ret[2] = { FLT_MAX, FLT_MAX };
//    int ret_cnt = 0;
//
//    if(intersections >= 1 && t0 >= 0 && t0 <= 1.0f){ ret[ret_cnt++] = t0; }
//    if(intersections >= 2 && t1 >= 0 && t1 <= 1.0f){ ret[ret_cnt++] = t1; }
//
//    t0 = ret[0];
//    t1 = ret[1];
//
//    return ret_cnt;
//}
//
//inline int RayConeIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 c_origin, SCI::Vex3 c_axis, float c_angle, float & t0, float & t1){
//    float cosAngle = cosf(c_angle);
//
//    //return __RayConeIntersection(r0,rd,c_origin,c_axis,cosAngle,t0,t1);
//    return __ConeSegmentIntersection(c_origin,c_axis,cosAngle,r0,r0+rd,t0,t1);
//}
//
//inline int RayConeIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 c_origin, SCI::Vex3 c_axis, SCI::Vex3 c_pnt, float & t0, float & t1){
//
//    float cosAngle = dot((c_pnt-c_origin).UnitVector(),c_axis.UnitVector());
//
//    //return __RayConeIntersection(r0,rd,c_origin,c_axis,cosAngle,t0,t1);
//    return __ConeSegmentIntersection(c_origin,c_axis,cosAngle,r0,r0+rd,t0,t1);
//}


inline SCI::Vex2 CalculateConic(SCI::Vex2 p0, SCI::Vex2 p1){
    SCI::Vex2 p0_2 = p0 * p0;
    SCI::Vex2 p1_2 = p1 * p1;
    float A, B;

    if(fabsf(p0.x-p1.x) > fabsf(p0.y-p1.y)){
        B = cross(p1_2,p0_2) / ( p1_2.x - p0_2.x );
        A = ( B * p0_2.x ) / ( B - p0_2.y );
    }
    else{
        A = cross(p0_2,p1_2) / ( p1_2.y - p0_2.y );
        B = ( A * p0_2.y ) / ( A - p0_2.x );
    }

    if(fabsf(p0.y-p1.y) < 0.00001f){    A = FLT_MAX;     }
    if(fabsf(p0.x-p1.x) < 0.00001f){    B = FLT_MAX;    }

    return SCI::Vex2(A,B);
}



//inline int ConicLineIntersection(SCI::Vex2 AB, SCI::Vex2 r0, SCI::Vex2 rd, float & t0, float & t1){
//
//    SCI::Vex2 R0 = r0*r0;
//    SCI::Vex2 RD = rd*rd;
//    SCI::Vex2 R0RD = r0*rd;
//
//    float _a = (AB.y*RD.x+AB.x*RD.y) / (AB.x*AB.y);
//    float _b = ((2.0f * AB.y * r0.x * rd.x) + (2.0f * AB.x * r0.y * rd.y)) / (AB.x*AB.y);
//    float _c = (AB.y * R0.x + AB.x * R0.y - AB.x*AB.y) / (AB.x*AB.y);
//
//    return QuadraticEquation(_a,_b,_c,t0,t1);
//}


inline bool PointInCone(SCI::Vex3 center, SCI::Vex3 _up, SCI::Vex3 _edge, SCI::Vex3 _pnt){

    SCI::Vex3 up = _up.UnitVector();
    SCI::Vex3 edge = _edge - center;
    SCI::Vex3 pnt = _pnt - center;

    float A = dot(up,edge)*dot(up,edge)/dot(edge,edge);


    return A <= dot(up,pnt)*dot(up,pnt)/dot(pnt,pnt);
}


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
inline SCI::Vex2  LinearBezier(float t, SCI::Vex2  p0, SCI::Vex2  p1){    return LinearBezier<SCI::Vex2>(t,p0,p1);  }
inline SCI::Vex3  LinearBezier(float t, SCI::Vex3  p0, SCI::Vex3  p1){    return LinearBezier<SCI::Vex3>(t,p0,p1);  }
inline SCI::Vex4  LinearBezier(float t, SCI::Vex4  p0, SCI::Vex4  p1){    return LinearBezier<SCI::Vex4>(t,p0,p1);  }


inline float QuadraticBezier(float t, float p0, float p1, float p2){ return QuadraticBezier<float>(t,p0,p1,p2); }
inline SCI::Vex2  QuadraticBezier(float t, SCI::Vex2  p0, SCI::Vex2  p1, SCI::Vex2  p2){ return QuadraticBezier<SCI::Vex2>(t,p0,p1,p2);  }
inline SCI::Vex3  QuadraticBezier(float t, SCI::Vex3  p0, SCI::Vex3  p1, SCI::Vex3  p2){ return QuadraticBezier<SCI::Vex3>(t,p0,p1,p2);  }
inline SCI::Vex4  QuadraticBezier(float t, SCI::Vex4  p0, SCI::Vex4  p1, SCI::Vex4  p2){ return QuadraticBezier<SCI::Vex4>(t,p0,p1,p2);  }

inline float CubicBezier(float t, float p0, float p1, float p2, float p3){ return CubicBezier<float>(t,p0,p1,p2,p3); }
inline SCI::Vex2  CubicBezier(float t, SCI::Vex2  p0, SCI::Vex2  p1, SCI::Vex2  p2, SCI::Vex2  p3){ return CubicBezier<SCI::Vex2>(t,p0,p1,p2,p3);  }
inline SCI::Vex3  CubicBezier(float t, SCI::Vex3  p0, SCI::Vex3  p1, SCI::Vex3  p2, SCI::Vex3  p3){ return CubicBezier<SCI::Vex3>(t,p0,p1,p2,p3);  }
inline SCI::Vex4  CubicBezier(float t, SCI::Vex4  p0, SCI::Vex4  p1, SCI::Vex4  p2, SCI::Vex4  p3){ return CubicBezier<SCI::Vex4>(t,p0,p1,p2,p3);  }

//inline SCI::Vex3 CubicBezier(float t, SCI::Vex3 p0, SCI::Vex3 p1, SCI::Vex3 p2, SCI::Vex3 p3){
//    float t0 = t;
//    float t1 = 1-t;
//    SCI::Vex3 ret;
//    ret += p0 * t1 * t1 * t1;
//    ret += p1 * t1 * t1 * t0 * 3.0f;
//    ret += p2 * t1 * t0 * t0 * 3.0f;
//    ret += p3 * t0 * t0 * t0;
//    return ret;
//}




inline float RayBoxIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 box_center, float box_size){

    SCI::Vex3 min_p = box_center - SCI::Vex3(box_size,box_size,box_size) / 2.0f;
    SCI::Vex3 max_p = box_center + SCI::Vex3(box_size,box_size,box_size) / 2.0f;

    float t = FLT_MAX;

    float tt[6];

    tt[0] = RayPlaneIntersection(r0,rd,SCI::Vex4(1,0,0,-min_p.x));
    tt[1] = RayPlaneIntersection(r0,rd,SCI::Vex4(1,0,0,-max_p.x));
    tt[2] = RayPlaneIntersection(r0,rd,SCI::Vex4(0,1,0,-min_p.y));
    tt[3] = RayPlaneIntersection(r0,rd,SCI::Vex4(0,1,0,-max_p.y));
    tt[4] = RayPlaneIntersection(r0,rd,SCI::Vex4(0,0,1,-min_p.z));
    tt[5] = RayPlaneIntersection(r0,rd,SCI::Vex4(0,0,1,-max_p.z));

    if(tt[0] > 0){
        SCI::Vex3 curp = r0 + rd * tt[0];
        if(curp.y >= min_p.y && curp.y <= max_p.y && curp.z >= min_p.z && curp.z <= max_p.z){
            t = SCI::Min(t,tt[0]);
        }
    }
    if(tt[1] > 0){
        SCI::Vex3 curp = r0 + rd * tt[1];
        if(curp.y >= min_p.y && curp.y <= max_p.y && curp.z >= min_p.z && curp.z <= max_p.z){
            t = SCI::Min(t,tt[1]);
        }
    }

    if(tt[2] > 0){
        SCI::Vex3 curp = r0 + rd * tt[2];
        if(curp.x >= min_p.x && curp.x <= max_p.x && curp.z >= min_p.z && curp.z <= max_p.z){
            t = SCI::Min(t,tt[2]);
        }
    }
    if(tt[3] > 0){
        SCI::Vex3 curp = r0 + rd * tt[3];
        if(curp.x >= min_p.x && curp.x <= max_p.x && curp.z >= min_p.z && curp.z <= max_p.z){
            t = SCI::Min(t,tt[3]);
        }
    }

    if(tt[4] > 0){
        SCI::Vex3 curp = r0 + rd * tt[4];
        if(curp.x >= min_p.x && curp.x <= max_p.x && curp.y >= min_p.y && curp.y <= max_p.y){
            t = SCI::Min(t,tt[4]);
        }
    }
    if(tt[5] > 0){
        SCI::Vex3 curp = r0 + rd * tt[5];
        if(curp.x >= min_p.x && curp.x <= max_p.x && curp.y >= min_p.y && curp.y <= max_p.y){
            t = SCI::Min(t,tt[5]);
        }
    }

    return t;
}


inline SCI::Vex3 RayQuadIntersection(SCI::Vex3 r0, SCI::Vex3 rd, SCI::Vex3 q[4]){
    SCI::Vex4 pln = CalculatePlane(q[0],q[1],q[2]);
    float t = RayPlaneIntersection(r0,rd,pln);
    if(t == FLT_MAX){ return SCI::VEX3_INVALID; }
    if(t < 0 || t > 1){ return SCI::VEX3_INVALID; }
    SCI::Vex3 pnt = r0 + rd * t;

    SCI::Vex3 n0 = -cross((pnt-q[0]).UnitVector(),(q[1]-q[0]).UnitVector());
    SCI::Vex3 n1 = -cross((pnt-q[1]).UnitVector(),(q[2]-q[1]).UnitVector());
    SCI::Vex3 n2 = -cross((pnt-q[2]).UnitVector(),(q[3]-q[2]).UnitVector());
    SCI::Vex3 n3 = -cross((pnt-q[3]).UnitVector(),(q[0]-q[3]).UnitVector());

    SCI::Vex4 f;

    f.x = SCI::dot(pln.xyz(),n0);
    f.y = SCI::dot(pln.xyz(),n1);
    f.z = SCI::dot(pln.xyz(),n2);
    f.w = SCI::dot(pln.xyz(),n3);

    if(f.x < 0 || f.y < 0 || f.z < 0 || f.w < 0){
        return SCI::VEX3_INVALID;
    }

    return pnt;
}


inline float LineBezierIntersection(SCI::Vex3 l0, SCI::Vex3 ld, SCI::Vex3 p0, SCI::Vex3 p1, SCI::Vex3 p2){
    float a = dot( (p0 + p2 - p1*2), ld );
    float b = dot( (p1 - p0), ld ) * 2;
    float c = dot( (p0 - l0), ld );

    float t0,t1;
    int sols = QuadraticEquation( a, b, c, t0, t1 );

    float ret = 0.5f;

    ret = (sols > 1 && t1 >= 0 && t1 <= 1) ? t1 : ret;
    ret = (sols > 0 && t0 >= 0 && t0 <= 1) ? t0 : ret;

    return ret;
}
