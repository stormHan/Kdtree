#ifndef _BASE_H_
#define _BASE_H_
#include "vector_types.h"
#include <string>

using namespace std;




// 短栈的大小
#define SHORT_STACK_SIZE 64

// 浮点数最大最小值
#define FLOAT_MAX (1.0E+10)
#define FLOAT_MIN (-1.0E+10)

#define LARGE_SMALL_SPLIT 64
#define LEAF_THRESHOLD 16

#define PI_  3.1415926

typedef float CUDA_FLOAT;

union FLOAT3
{
    struct{float x, y, z;};
	float3 f3;
	float f[3];
};

union FLOAT4
{
    struct{float x, y, z, w;};
	float4 f4;
	float f[4];
};


typedef FLOAT4 VertexType;
typedef int4  FaceType;
typedef FLOAT4 TextureType;
typedef uint4  FaceVTType;
typedef FLOAT4 NormalType;
typedef uint4  FaceVNType;


typedef struct _tri
{
	float x;
	float y;
	float z;
}Tri;


#define KEY_MAX 300

struct SceneInfo
{
	// face
    size_t vertexNum;
	size_t faceNum;
	VertexType *h_vertex;
	FaceType *h_face;

	// texture
	size_t textureNum;
	size_t faceTextureNum;
	TextureType *h_texture;
	FaceVTType *h_facetexture;


	// normal
	size_t normalNum;
	size_t faceNormalNum;
	NormalType *h_normal;
	FaceVNType  *h_facenormal;
	
};

struct SceneInfoArr
{
    // face
	size_t vertexNum[KEY_MAX];
	size_t faceNum[KEY_MAX];
	VertexType *h_vertex[KEY_MAX];
	FaceType *h_face[KEY_MAX];

	// texture
	size_t textureNum[KEY_MAX];
	size_t faceTextureNum[KEY_MAX];
	TextureType *h_texture[KEY_MAX];
	FaceVTType *h_facetexture[KEY_MAX];

	// normal
	size_t normalNum[KEY_MAX];
	size_t faceNormalNum[KEY_MAX];
	NormalType *h_normal[KEY_MAX];
	FaceVNType *h_facenormal[KEY_MAX];

};

//// mtl
struct Mtl
{
	string name;
	FLOAT4 Ka;
	FLOAT4 Kd;
	FLOAT4 Ks;
	float Tr;
	float Ns;
    char map_Kd[100];
	size_t wid;
	size_t hei;
	bool isMapKd;
	// uchar4 *texture;
	int texIndex;
};

////
struct Tex
{
	//string name;
	size_t wid;
	size_t hei;
	char map_Kd[100];
	uchar4 *texPtr;
};

#define TEXMAX 100
#define TEX_G_WID 1500
#define TEX_G_HEI 20000

#define TIGHT_CYCLE 4

//// 渲染开关
#define NEED_NORMAL
#define NEED_TEXTURE
#define NEED_LIGHT
#define NEED_BUMP_TEXTURE

#define BB_EXTENT (0.01)

// 测试开关
//#define NEED_TIMER

#ifdef NEED_TIMER
#define TIMER_START(a) CuTimer a;a.startTimer()
#define TIMER_FINISH(a, b) a.finishTimer(b)
#else
#define TIMER_START(a) 
#define TIMER_FINISH(a, b)
#endif

#define SMALL_HANDLE_MAX 3


#define BLOCK_SIZE 64
#define HANDLE_NUM 1


// gpu kd
#define CREATE_ROOT_BLOCK_SIZE BLOCK_SIZE
#define CREATE_ROOT_THREAD_HANDLE_NUM HANDLE_NUM

#define CAL_PRIMS_BB_BLOCK_SIZE BLOCK_SIZE
#define CAL_PRIMS_BB_THREAD_HANDLE_NUM HANDLE_NUM

#define COLLECT_TRINODE_BB_BLOCK_SIZE BLOCK_SIZE
#define COLLECT_TRINODE_BB_THREAD_HANDLE_NUM HANDLE_NUM

#define SET_ACTIVE_NODE_BB_BLOCK_SIZE BLOCK_SIZE
#define SET_ACTIVE_NODE_BB_THREAD_HANDLE_NUM HANDLE_NUM

#define SET_DEVICE_MEM_BLOCK_SIZE BLOCK_SIZE
#define SET_DEVICE_MEM_THREAD_HANDLE_NUM HANDLE_NUM

#define SET_FLAG_WITH_SPLIT_BLOCK_SIZE BLOCK_SIZE
#define SET_FLAG_WITH_SPLIT_THREAD_HANDLE_NUM HANDLE_NUM

#define FINISH_NEXT_LIST_BLOCK_SIZE BLOCK_SIZE
#define FINISH_NEXT_LIST_THREAD_HANDLE_NUM HANDLE_NUM

#define FILTER_FINISH_BLOCK_SIZE BLOCK_SIZE
#define FILTER_FINISH_THREAD_HANDLE_NUM HANDLE_NUM

#define SET_SMALL_NODE_MASK_BLOCK_SIZE BLOCK_SIZE
#define SET_SMALL_NODE_MASK_THREAD_HANDLE_NUM HANDLE_NUM

#define INIT_BOUNDRY_BLOCK_SIZE BLOCK_SIZE
#define INIT_BOUNDRY_THREAD_HANDLE_NUM HANDLE_NUM

#define SET_LEFT_RIGHT_MASK_BLOCK_SIZE BLOCK_SIZE
#define SET_LEFT_RIGHT_MASK_THREAD_HANDLE_NUM HANDLE_NUM

#define PROCESS_SMALL_NODE_BLOCK_SIZE BLOCK_SIZE
#define PROCESS_SMALL_NODE_THREAD_HANDLE_NUM HANDLE_NUM

#define LEAF_FILTER_BLOCK_SIZE BLOCK_SIZE
#define LEAF_FILTER_THREAD_HANDLE_NUM HANDLE_NUM

#endif