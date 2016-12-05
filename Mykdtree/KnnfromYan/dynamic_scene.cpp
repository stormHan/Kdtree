#include "vector_types.h"
#include <math.h>
//void nor_(float4& vec)
//{
//    float lenInv = 1.0f / sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
//	float4 newVec;
//	newVec.x = vec.x * lenInv;
//	newVec.y = vec.y * lenInv;
//	newVec.z = vec.z * lenInv;
//
//	vec = newVec;
//}
//
//
//float4 cross_(const float4& v1, const float4& v2)
//{
//	float4 v3;
//	v3.x = v1.y * v2.z - v1.z * v2.y;
//	v3.y = v1.z * v2.x - v1.x * v2.z;
//	v3.z = v1.x * v2.y - v1.y * v2.x;
//
//	return v3;
//}
//
//
//float dot_(const float4 v1, const float4 v2)
//{
//	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
//}
//
//
//float4 sub_(const float4& v1, const float4& v2)
//{
//    float4 v3;
//	v3.x = v1.x - v2.x;
//	v3.y = v1.y - v2.y;
//	v3.z = v1.z - v2.z;
//	return v3;
//}
