#ifndef _KDGPU_DATA_H_
#define _KDGPU_DATA_H_
#include "vector_types.h"

//#include "RayTrace/Workspace.h"
#include "cudpp.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "base.h"


// bezier 曲面控制点数据
struct Prims
{
	// ****************成员数据******************
  
	// face
	size_t vertexNum_[KEY_MAX];
	size_t faceNum_[KEY_MAX];
	VertexType *h_vertex_[KEY_MAX];
	FaceType *h_face_[KEY_MAX];
	VertexType *d_vertex_[KEY_MAX];
	FaceType *d_face_[KEY_MAX];


	// texture
	size_t textureNum_[KEY_MAX];
	size_t faceTextureNum_[KEY_MAX];
	TextureType *h_texture_[KEY_MAX];
	FaceVTType *h_facetexture_[KEY_MAX];
	TextureType *d_texture_[KEY_MAX];
	FaceVTType *d_facetexture_[KEY_MAX];



	// normal
	size_t normalNum_[KEY_MAX];
	size_t faceNormalNum_[KEY_MAX];
	NormalType *h_normal_[KEY_MAX];
	FaceVNType *h_facenormal_[KEY_MAX];
	NormalType *d_normal_[KEY_MAX];
	FaceVNType *d_facenormal_[KEY_MAX];

	void initPrims(SceneInfoArr &scene, int ppmNum);


};

// *******************Bounding Box **********
struct BoundingBox
{
    FLOAT4 bbMin_;	//!< @internal the min boundry of bb
	FLOAT4 bbMax_;   //!< @internal the max boundry of bb
};

// every primitive's bb
struct AllBoundingBoxList
{
	size_t size_;
    float4 *primsBoundingBox_[2];

	void initPrimsBoundingBox(size_t nPrims);
	void realloc(size_t referSize, bool needCpy);

	
};

// ***************tri-node list bb list
struct TriNodePrimsBBList
{
 //   size_t capacity_;    // may be > nPrims
	size_t size_;
	//size_t *d_capacity_;
	//size_t *d_size_;
	// minx, miny, minz, maxx, maxy, maxz
	float *triNodePrimsBBList_[6];
	
	void initTriNodePrimsBBList(size_t capacity);
	void realloc(size_t referSize, bool needCpy);

	


};

// ************** kd-tree node(origin version)*******
struct KdNodeOrigin
{
	//size_t flags_;				// 标记该node是其父亲节点的左孩子，还是右孩子
	size_t splitType_;			// split的类型，内部节点为x, y, z, 外部节点leaf
    size_t addrTriNodeList_;    // 指向所属triNodeList的起始地址
	size_t nTris_;				// 该node有多少个tri
	float splitValue_;			// split的值
	BoundingBox bb_;			// 该node的包围盒
	int lChild_;//KdNodeOrigin *lChild_;		// node的左孩子节点指针
	int rChild_;//KdNodeOrigin *rChild_;      // node的右孩子节点指针
	//KdNodeOrigin *father_;		// node的父亲节点指针
};

struct KdNode_base
{
	size_t splitType_;
	float splitValue_;
	int lChild_;
	int rChild_;
};

struct KdNode_bb
{
	BoundingBox bb_;
};

struct KdNode_extra
{
	size_t addrTriNodeList_;
	size_t nTris_;
};

// ***************node list****************
template <typename T, int dimension>
struct  
{
    T *list_[dimension];					//!< @internal list
	size_t size_;
	size_t capacity_;

	void initList(size_t num);
	void realloc(size_t referSize, bool needCpy, bool needFree = true);
	void release();
 
	
};

template <typename T, int dimension>
void List<T, dimension>::initList(size_t num)
{
	size_ = 0;
	capacity_ = num;

	for (int i = 0; i < dimension; i++)
	{
	    cudaMalloc((void**)&list_[i], sizeof(T) * capacity_);
	}
    
}

template <typename T, int dimension>
void List<T, dimension>::realloc(size_t referSize, bool needCpy, bool needFree = true)
{
	// needn't realloc mem
	if (referSize <= capacity_) return;

	// save old size and mem
	T *oldMem[dimension];
	for (int i = 0; i < dimension; i++)
	{
	    oldMem[i] = list_[i];
	}
	//size_t oldSize = size_;

	//  cal new size
    while(capacity_ < referSize)
	{
	    capacity_ <<= 1;
	}
	
	// alloc new mem
	for (int i = 0; i < dimension; i++)
	{
	    cudaMalloc((void**)&list_[i], sizeof(T) * capacity_);
	}
	

	// memcpy
	if (needCpy)
	{
		for (int i = 0; i < dimension; i++)
		{
		    cudaMemcpy(list_[i], oldMem[i], sizeof(T) * size_, cudaMemcpyDeviceToDevice);
		}
	}

	if (needFree)
	{
	    // release old mem
		for (int i = 0; i < dimension; i++)
		{
			cudaFree(oldMem[i]);
		}
	}
	
	
	

}
template <typename T, int dimension>
void List<T, dimension>::release()
{
    for (int i = 0; i < dimension; i++)
	{
		if (list_[i] != NULL)
	        CUDA_SAFE_CALL(cudaFree(list_[i]));
	}

	size_ = 0;
}


typedef List<KdNodeOrigin, 1> NodeList;		// node list store nodes which have handled
typedef List<KdNodeOrigin, 1> ActiveList;	// active list store nodes which will be handling
typedef List<KdNodeOrigin, 1> NextList;		// next list store nodes which will be handle in next loop
typedef List<KdNodeOrigin, 1> SmallList;     // small list store small node



// ***************tri-node list******************
struct TriNodeList
{	

    size_t* triIdx_;		//!< @internal triangle index
	size_t* segFlags_;		//!< @internal segment flags
	
	size_t  size_;

	void initTriNodeList(size_t num);
	void realloc(size_t referSize, bool needCpy);

	
};









#endif