#ifndef _DYNAMIC_SCENE_H_
#define _DYNAMIC_SCENE_H_
#include "vector_types.h"
#include "vector_utility.h"

//void nor_(float4& vec);
//float4 cross_(const float4& v1, const float4& v2);
//float dot_(const float4 v1, const float4 v2);
//float4 sub_(const float4& v1, const float4& v2);

// 光线
struct Ray_d
{
    FLOAT4 org_;
	FLOAT4 dir_;
	CUDA_FLOAT  tMin_;
	CUDA_FLOAT  tMax_;
};

// 点光源
struct PointLight_d
{
    FLOAT4 pos_;
	FLOAT4 color_;
};

// 相机参数
struct Camera_d
{
	FLOAT4 from_;
	FLOAT4 to_;
	FLOAT4 up_;

	CUDA_FLOAT  near_;
	CUDA_FLOAT  far_;

	CUDA_FLOAT  fovy_;
	CUDA_FLOAT  aspect_;

	//
	FLOAT4 xDir_;
	FLOAT4 yDir_;
	FLOAT4 zDir_;

	void calCamera()
	{
	    zDir_ = sub4(to_, from_);
		nor4(zDir_);
		xDir_ = cross4(zDir_, up_);
		nor4(xDir_);
		yDir_ = cross4(xDir_, zDir_);
		nor4(yDir_);
	}
};

// 视平面
struct View_d
{
    int wid_;
	int hei_;
	
	int samplePerPixel_;
};

// 场景参数
struct Scene_d
{
    // 点光源
	PointLight_d pointLight_;
	
	// 照相机
	Camera_d camera_;

	// 视平面
	View_d view_;
};

//typedef struct SCENE
//{
//	float4 org_;
//	float4 dir_;
//	float  tMin_;
//	float  tMax_;
//
//	float4 from_;
//	float4 to_;
//	float4 up_;
//
//	float  near_;
//	float  far_;
//
//	float  fovy_;
//	float  aspect_;
//
//	//
//	float4 xDir_;
//	float4 yDir_;
//	float4 zDir_;
//
//	int wid_;
//	int hei_;
//	
//	int samplePerPixel_;
//
//
//    
//}SCENE_;

#endif
