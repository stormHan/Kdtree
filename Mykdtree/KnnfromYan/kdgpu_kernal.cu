#ifndef _KDGPU_KERNAL_CU_
#define _KDGPU_KERNAL_CU_
#include "vector_types.h"
#include "kdgpu_data.h"
#include "base.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <float.h>

#ifdef NEED_TEXTURE
__constant__ uint4 d_texPos[TEXMAX];
// __constant__ uint2 d_texWidHei[TEXMAX];

//// texture map
texture<uchar4, 2, cudaReadModeElementType>texKa;
//texture<uchar4, 2, cudaReadModeElementType>texKd0;
//texture<uchar4, 2, cudaReadModeElementType>texKd1;
//texture<uchar4, 2, cudaReadModeElementType>texKd2;
//texture<uchar4, 2, cudaReadModeElementType>texKd3;
//texture<uchar4, 2, cudaReadModeElementType>texKd4;
//texture<uchar4, 2, cudaReadModeElementType>texKd5;
//texture<uchar4, 2, cudaReadModeElementType>texKd6;
//texture<uchar4, 2, cudaReadModeElementType>texKd7;
//texture<uchar4, 2, cudaReadModeElementType>texKd8;
//texture<uchar4, 2, cudaReadModeElementType>texKd9;
//texture<uchar4, 2, cudaReadModeElementType>texKd10;
//texture<uchar4, 2, cudaReadModeElementType>texKd11;
//texture<uchar4, 2, cudaReadModeElementType>texKd12;
//texture<uchar4, 2, cudaReadModeElementType>texKd13;
//texture<uchar4, 2, cudaReadModeElementType>texKd14;
//texture<uchar4, 2, cudaReadModeElementType>texKd15;
//texture<uchar4, 2, cudaReadModeElementType>texKd16;
//texture<uchar4, 2, cudaReadModeElementType>texKd17;
//texture<uchar4, 2, cudaReadModeElementType>texKd18;
//texture<uchar4, 2, cudaReadModeElementType>texKd19;
//texture<uchar4, 2, cudaReadModeElementType>texKd20;
//texture<uchar4, 2, cudaReadModeElementType>texKd21;
//texture<uchar4, 2, cudaReadModeElementType>texKd22;
//texture<uchar4, 2, cudaReadModeElementType>texKd23;
//texture<uchar4, 2, cudaReadModeElementType>texKd24;
//texture<uchar4, 2, cudaReadModeElementType>texKd25;
//texture<uchar4, 2, cudaReadModeElementType>texKd26;
//texture<uchar4, 2, cudaReadModeElementType>texKd27;
//texture<uchar4, 2, cudaReadModeElementType>texKd28;
//texture<uchar4, 2, cudaReadModeElementType>texKd29;
//texture<uchar4, 2, cudaReadModeElementType>texKd30;
#endif

//// mask
__constant__ size_t d_rootMaskHi_[64];
__constant__ size_t d_rootMaskLo_[64];
__constant__ size_t d_rootStartMaskHi_[64];
__constant__ size_t d_rootStartMaskLo_[64];
__constant__ size_t d_rootEndMaskHi_[64];
__constant__ size_t d_rootEndMaskLo_[64];

__constant__ char d_bits_in_char[256];
__constant__ char d_one_loc[2048];


__device__ 
inline size_t fastBitCounting64(size_t hi, size_t lo)
{
	return   d_bits_in_char [hi         & 0xffu]
	+  d_bits_in_char [(hi >>  8) & 0xffu]
	+  d_bits_in_char [(hi >> 16) & 0xffu]
	+  d_bits_in_char [(hi >> 24) & 0xffu] 
	+  d_bits_in_char [lo         & 0xffu]
	+  d_bits_in_char [(lo >>  8) & 0xffu]
	+  d_bits_in_char [(lo >> 16) & 0xffu]
	+  d_bits_in_char [(lo >> 24) & 0xffu];
}

__device__
inline size_t fastBitCounting32(size_t n)
{
	return   d_bits_in_char [n         & 0xffu]
	+  d_bits_in_char [(n >>  8) & 0xffu]
	+  d_bits_in_char [(n >> 16) & 0xffu]
	+  d_bits_in_char [(n >> 24) & 0xffu];
}


// 测试纹理绑定
#ifdef NEED_TEXTURE
__global__
void testTextureBind(size_t wid_, size_t hei_, uchar4* d_tex, int texIndex)
{
	/*size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
	size_t tidy = blockDim.y * blockIdx.y + threadIdx.y;*/

	//if (tidx < wid_ && tidy < hei_)
	//{
	//    float fx = tidx * 1. / wid_;
	//	float fy = tidy * 1. / hei_;

	//	int index = tidy * wid_ + tidx;
	//	d_tex[index] = tex2D(texKd0, fx, fy);
	//	/*int index = tidy * wid_ + tidx;
	//	d_tex[index] = tex2D(texKd0, tidx, tidy);*/
	//}

	//size_t pixelNum = wid_ * hei_;
	//size_t tid = threadIdx.x;
	//size_t bid = blockIdx.x;
	//size_t id = bid * blockDim.x + tid;

	/*if (id < pixelNum)
	{
		switch(texIndex)
		{
		case 0:
			d_tex[id] = tex1Dfetch(texKd0, id);
			break;
		case 1:
			d_tex[id] = tex1Dfetch(texKd1, id);
			break;
		case 2:
			d_tex[id] = tex1Dfetch(texKd2, id);
			break;
		case 3:
			d_tex[id] = tex1Dfetch(texKd3, id);
			break;
		case 4:
			d_tex[id] = tex1Dfetch(texKd4, id);
			break;
		case 5:
			d_tex[id] = tex1Dfetch(texKd5, id);
			break;
		case 6:
			d_tex[id] = tex1Dfetch(texKd6, id);
			break;
		case 7:
			d_tex[id] = tex1Dfetch(texKd7, id);
			break;
		case 8:
			d_tex[id] = tex1Dfetch(texKd8, id);
			break;
		case 9:
			d_tex[id] = tex1Dfetch(texKd9, id);
			break;
		case 10:
			d_tex[id] = tex1Dfetch(texKd10, id);
			break;
		case 11:
			d_tex[id] = tex1Dfetch(texKd11, id);
			break;
		default:
			break;
		}*/
	 
	//}
}

__global__
void fetchTextureKernalTest(uchar4* d_tex, int w, int h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = idy * w + idx;
	if (idx < w && idy < h)
		d_tex[index] = tex2D(texKa, idx, idy);

	/*float fdx = idx * 1. / wid;
	float fdy = idy * 1. / hei;
	if (idx < wid && idy < hei)
	{
	    d_tex[index] = tex2D(tex, fdx, fdy);
	}*/
}
#endif


// 创建根节点
__global__ 
void createRootKernal(size_t		nPrims,
					  int *activeList,
					  //~KdNodeOrigin  *kdNode,
					  KdNode_base   *kdNodeBase,
					  KdNode_bb     *kdNodeBB,
					  KdNode_extra  *kdNodeExtra,
					  size_t		*triIdx,
					  size_t		*segFlags)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;

	size_t idx = bidx * blockDim.x + tidx;

	size_t everyThreadNum = CREATE_ROOT_THREAD_HANDLE_NUM;
	size_t threadNum = CREATE_ROOT_BLOCK_SIZE * gridDim.x;

	
	// init tri-node list
	size_t addrIdx;
	size_t *triAddr = NULL;

	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = threadNum * i + idx;
		triAddr = triIdx + addrIdx;
		if (addrIdx < nPrims)
		{
			*triAddr = addrIdx;		//*triAddr = idx; 2010 5.24 9.30
		}

	}
	if (0 == idx) 
	{
	    BoundingBox newBB;
		newBB.bbMin_.x = FLOAT_MIN;
		newBB.bbMin_.y = FLOAT_MIN;
		newBB.bbMin_.z = FLOAT_MIN;
		newBB.bbMax_.x = FLOAT_MAX;
		newBB.bbMax_.y = FLOAT_MAX;
		newBB.bbMax_.z = FLOAT_MAX;

		// create root 
		//~kdNode->addrTriNodeList_ = 0;
		//~kdNode->nTris_	= nPrims;
		//~kdNode->bb_		= newBB;
		//~kdNode->lChild_ = -1;
		//~kdNode->rChild_ = -1;
		kdNodeExtra->addrTriNodeList_ = 0;
		kdNodeExtra->nTris_ = nPrims;
		kdNodeBB->bb_ = newBB;
		kdNodeBase->lChild_ = -1;
		kdNodeBase->rChild_ = -1;

		activeList[0]	= 0;
		
		*segFlags = 1;

	}
}

// 计算每个三角面片的包围盒
__global__ 
void calPrimsBoundingBoxKernal(	size_t nPrims, 
								// float4 *primsBoundingBox[], 
								float4 *primsBoundingBoxMin,
								float4 *primsBoundingBoxMax,
								//float4 *d_iPrims[]
								///*float4 *prims0,
								//float4 *prims1,
								//float4 *prims2*/
								VertexType *vertex,
								FaceType *face
							   )
{
	/*size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;*/
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float4 *primsBoundingBox[2];

	if (threadIdx.x == 0){
	primsBoundingBox[0] = primsBoundingBoxMin;
	primsBoundingBox[1] = primsBoundingBoxMax;
	}
	__syncthreads();

	/*float4 *d_iPrims[3];
	d_iPrims[0] = prims0;
	d_iPrims[1] = prims1;
	d_iPrims[2] = prims2;*/

	size_t everyThreadNum = CAL_PRIMS_BB_THREAD_HANDLE_NUM;
	size_t threadNum = CAL_PRIMS_BB_BLOCK_SIZE * gridDim.x;

	// cal bounding box
	size_t addrIdx;
	/*float4 triCtrlPoints[3];*/
	
	float4 bbMin;
	float4 bbMax;
	FLOAT4 ctrlPoint;
	__shared__ FLOAT4 triCtrlPoints[CAL_PRIMS_BB_BLOCK_SIZE][3];
	//__shared__ FLOAT4 bbMin[CAL_PRIMS_BB_BLOCK_SIZE];
	//__shared__ FLOAT4 bbMax[CAL_PRIMS_BB_BLOCK_SIZE];
	//__shared__ FLOAT4 ctrlPoint[CAL_PRIMS_BB_BLOCK_SIZE];
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < nPrims)
		{
			// fetch the bezier patch's all ctrl points
			/*for (size_t j = 0; j < 3; j++)
			{
				triCtrlPoints[j] = d_iPrims[j][addrIdx];
			}*/
			FaceType newFace = face[addrIdx];
			/*triCtrlPoints[0] = vertex[newFace.x].f4;
			triCtrlPoints[1] = vertex[newFace.y].f4;
			triCtrlPoints[2] = vertex[newFace.z].f4;*/
			triCtrlPoints[threadIdx.x][0].f4 = vertex[newFace.x].f4;
			triCtrlPoints[threadIdx.x][1].f4 = vertex[newFace.y].f4;
			triCtrlPoints[threadIdx.x][2].f4 = vertex[newFace.z].f4;
			__syncthreads();


			// cal bb
			bbMin.x = bbMin.y = bbMin.z = FLOAT_MAX;
			bbMax.x = bbMax.y = bbMax.z = FLOAT_MIN;
			for (size_t j = 0; j < 3; j++)
			{
				ctrlPoint = triCtrlPoints[threadIdx.x][j];

				bbMin.x = (ctrlPoint.x < bbMin.x) ? ctrlPoint.x : bbMin.x;
				bbMax.x = (ctrlPoint.x > bbMax.x) ? ctrlPoint.x : bbMax.x;
				bbMin.y = (ctrlPoint.y < bbMin.y) ? ctrlPoint.y : bbMin.y;
				bbMax.y = (ctrlPoint.y > bbMax.y) ? ctrlPoint.y : bbMax.y;
				bbMin.z = (ctrlPoint.z < bbMin.z) ? ctrlPoint.z : bbMin.z;
				bbMax.z = (ctrlPoint.z > bbMax.z) ? ctrlPoint.z : bbMax.z;
			} 

			if (bbMin.x == bbMax.x)
			{
				bbMin.x -= BB_EXTENT;
				bbMax.x += BB_EXTENT;
			}

			if (bbMin.y == bbMax.y)
			{
				bbMin.y -= BB_EXTENT;
				bbMax.y += BB_EXTENT;
			}

			if (bbMin.z == bbMax.z)
			{
				bbMin.z -= BB_EXTENT;
				bbMax.z += BB_EXTENT;
			}

			// write to the primsBoundingboxList
			primsBoundingBox[0][addrIdx] = bbMin;
			primsBoundingBox[1][addrIdx] = bbMax;
		}
	}
}

// 收集tri-node list 中的prims的bb
__global__
void collectTriNodePrimsBBKernal(	//float4  **primsBoundingBox,
								    float4  *primsBoundingBoxMin,
									float4  *primsBoundingBoxMax,
									//float   **triNodePrimsBBList,
									float   *triNodePrimsBBList0,
									float   *triNodePrimsBBList1,
									float   *triNodePrimsBBList2,
									float   *triNodePrimsBBList3,
									float   *triNodePrimsBBList4,
									float   *triNodePrimsBBList5,

									size_t  *triIdx,
									size_t  triNodeSize)
{
	// collect
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;

	// 
	float4 *primsBoundingBox[2];
	primsBoundingBox[0] = primsBoundingBoxMin;
	primsBoundingBox[1] = primsBoundingBoxMax;
	float *triNodePrimsBBList[6];
	triNodePrimsBBList[0] = triNodePrimsBBList0;
	triNodePrimsBBList[1] = triNodePrimsBBList1;
	triNodePrimsBBList[2] = triNodePrimsBBList2;
	triNodePrimsBBList[3] = triNodePrimsBBList3;
	triNodePrimsBBList[4] = triNodePrimsBBList4;
	triNodePrimsBBList[5] = triNodePrimsBBList5;


	size_t everyThreadNum = COLLECT_TRINODE_BB_THREAD_HANDLE_NUM;
	size_t threadNum = COLLECT_TRINODE_BB_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;
	float4 tempMin, tempMax;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < triNodeSize)
		{
			tempMin = primsBoundingBox[0][triIdx[addrIdx]];
			tempMax = primsBoundingBox[1][triIdx[addrIdx]];
			triNodePrimsBBList[0][addrIdx] = tempMin.x;
			triNodePrimsBBList[1][addrIdx] = tempMin.y;
			triNodePrimsBBList[2][addrIdx] = tempMin.z;
			triNodePrimsBBList[3][addrIdx] = tempMax.x;
			triNodePrimsBBList[4][addrIdx] = tempMax.y;
			triNodePrimsBBList[5][addrIdx] = tempMax.z;
		}
	}
}

// 包围盒的与操作
__device__
BoundingBox tightBB(BoundingBox &newBB, BoundingBox &oldBB)
{
	BoundingBox retBB;

	retBB.bbMin_.x = (newBB.bbMin_.x >= oldBB.bbMin_.x) ? newBB.bbMin_.x : oldBB.bbMin_.x;
	retBB.bbMin_.y = (newBB.bbMin_.y >= oldBB.bbMin_.y) ? newBB.bbMin_.y : oldBB.bbMin_.y;
	retBB.bbMin_.z = (newBB.bbMin_.z >= oldBB.bbMin_.z) ? newBB.bbMin_.z : oldBB.bbMin_.z;

	retBB.bbMax_.x = (newBB.bbMax_.x <= oldBB.bbMax_.x) ? newBB.bbMax_.x : oldBB.bbMax_.x;
	retBB.bbMax_.y = (newBB.bbMax_.y <= oldBB.bbMax_.y) ? newBB.bbMax_.y : oldBB.bbMax_.y;
	retBB.bbMax_.z = (newBB.bbMax_.z <= oldBB.bbMax_.z) ? newBB.bbMax_.z : oldBB.bbMax_.z;

	return retBB;
}

// 设置activeNode中每个node的boundingBox,splittype_, splitvalue
// 并且填充左右孩子节点的一些必要内容，father, flag
__global__ 
void set_ActiveNode_Child_PreDistribute_Kernal(  //float			*d_odata[], 
											   float        *d_odata0,
											   float        *d_odata1,
											   float        *d_odata2,
											   float        *d_odata3,
											   float        *d_odata4,
											   float        *d_odata5,

											   int *activeList,
											   int *nextList,
											   //~KdNodeOrigin   *kdNode,
											   KdNode_base    *kdNodeBase,
											   KdNode_bb      *kdNodeBB,
											   KdNode_extra   *kdNodeExtra,
											   size_t         kdSize,
											   size_t			activeListSize,
											   float			*splitList,
											   size_t         isNeedTight,
											   size_t         *splitTypeList)
{
	// collect
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;

	//
	float *d_odata[6];
	d_odata[0] = d_odata0;
	d_odata[1] = d_odata1;
	d_odata[2] = d_odata2;
	d_odata[3] = d_odata3;
	d_odata[4] = d_odata4;
	d_odata[5] = d_odata5;


	size_t everyThreadNum = SET_ACTIVE_NODE_BB_THREAD_HANDLE_NUM;
	size_t threadNum = SET_ACTIVE_NODE_BB_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	// setting
	int activeNode;
	int lChild, rChild;
	size_t addrTriNode;
	BoundingBox newBB, realBB;
	float3 range;
	float splitValue;
	size_t splitType;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < activeListSize)
		{
			// 得到当前active node 的指针
			activeNode = activeList[addrIdx];

			// 当前active node相关的triNodeList的开始地址
			//~addrTriNode = kdNode[activeNode].addrTriNodeList_;
			addrTriNode = kdNodeExtra[activeNode].addrTriNodeList_;

			//~realBB = kdNode[activeNode].bb_;
			realBB = kdNodeBB[activeNode].bb_;
			/*if (isNeedTight)
			{*/
				// 获取node的包围盒
				newBB.bbMin_.x = d_odata[0][addrTriNode];
				newBB.bbMin_.y = d_odata[1][addrTriNode];
				newBB.bbMin_.z = d_odata[2][addrTriNode];
				newBB.bbMax_.x = d_odata[3][addrTriNode];
				newBB.bbMax_.y = d_odata[4][addrTriNode];
				newBB.bbMax_.z = d_odata[5][addrTriNode];

				// 紧缩包围盒
				realBB = tightBB(newBB, realBB);
				//~kdNode[activeNode].bb_ = realBB;
				kdNodeBB[activeNode].bb_ = realBB;
			//}
			
			

			// 执行split操作，得到leftChild, rightChild
			lChild = kdSize + addrIdx;
			rChild = kdSize + activeListSize + addrIdx;
			//~kdNode[activeNode].lChild_ = lChild;
			//~kdNode[activeNode].rChild_ = rChild;
			kdNodeBase[activeNode].lChild_ = lChild;
			kdNodeBase[activeNode].rChild_ = rChild;
			// init nextList
			nextList[addrIdx] = lChild;
			nextList[addrIdx + activeListSize] = rChild;

			// 确定split在哪一个轴上进行
			range.x = realBB.bbMax_.x - realBB.bbMin_.x;
			range.y = realBB.bbMax_.y - realBB.bbMin_.y;
			range.z = realBB.bbMax_.z - realBB.bbMin_.z;
			splitType = (range.x > range.y) ? ((range.x > range.z)? 1 : 3) : ((range.y > range.z)? 2 : 3); 
			//~kdNode[activeNode].splitType_ = splitType;
			kdNodeBase[activeNode].splitType_ = splitType;

			// 计算split value	
			BoundingBox leftBB = realBB;
			BoundingBox rightBB = realBB;

			//KdNode_base newKdNodeBase;
			//~switch(kdNode[activeNode].splitType_)
			switch(kdNodeBase[activeNode].splitType_)
			{
			case 1:
				splitValue = (realBB.bbMax_.x + realBB.bbMin_.x) / 2;
				//~kdNode[activeNode].splitValue_ = splitValue;
				kdNodeBase[activeNode].splitValue_ = splitValue;

				// set child bb
				leftBB.bbMax_.x = splitValue;
				rightBB.bbMin_.x = splitValue;
				//~kdNode[kdNode[activeNode].lChild_].bb_ = leftBB;
				//~kdNode[kdNode[activeNode].rChild_].bb_ = rightBB;

				kdNodeBB[lChild].bb_ = leftBB;
				kdNodeBB[rChild].bb_ = rightBB;
				break;
			case 2:
				splitValue = (realBB.bbMax_.y + realBB.bbMin_.y) / 2;
				//~kdNode[activeNode].splitValue_ = splitValue;
				kdNodeBase[activeNode].splitValue_ = splitValue;

				// set child bb
				leftBB.bbMax_.y = splitValue;
				rightBB.bbMin_.y = splitValue;
				
				//~kdNode[kdNode[activeNode].lChild_].bb_ = leftBB;
				//~kdNode[kdNode[activeNode].rChild_].bb_ = rightBB
				kdNodeBB[lChild].bb_ = leftBB;
				kdNodeBB[rChild].bb_ = rightBB;
				break;
			case 3:
				splitValue = (realBB.bbMax_.z + realBB.bbMin_.z) / 2;
				//~kdNode[activeNode].splitValue_ = splitValue;
				kdNodeBase[activeNode].splitValue_ = splitValue;

				// set child bb
				leftBB.bbMax_.z = splitValue;
				rightBB.bbMin_.z = splitValue;
				//~kdNode[kdNode[activeNode].lChild_].bb_ = leftBB;
				//~kdNode[kdNode[activeNode].rChild_].bb_ = rightBB;
				kdNodeBB[lChild].bb_ = leftBB;
				kdNodeBB[rChild].bb_ = rightBB;
				break;
			default:
				break;
			}
			

			// 设置ditribute的segment开始值
			splitList[addrTriNode] = splitValue;
			splitTypeList[addrTriNode] = splitType;	
		}
	}
}




// 设置device一段连续空间的值
template <typename T>
__global__
void deviceMemsetKernal(T *d_idata, T value, size_t size)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;

	size_t everyThreadNum = CAL_PRIMS_BB_THREAD_HANDLE_NUM;
	size_t threadNum = CAL_PRIMS_BB_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < size)
		{
			d_idata[addrIdx] = value;
		}
	}
}

// 划分左右孩子
__global__
void setFlagWithSplitKernal(//float4     **primsBoundingBox,
							float4     *primsBoundingBoxMin,
							float4     *primsBoundingBoxMax,

							size_t	   *triIdx,
							size_t     *d_ileftChildListFlagsAssist,
							size_t     *d_irightChildListFlagsAssist,
							float      *d_osplitListAssist,
							size_t     *d_osplitTypeListAssist,
							size_t     triNodeListSize)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	//
	float4 *primsBoundingBox[2];
	primsBoundingBox[0] = primsBoundingBoxMin;
	primsBoundingBox[1] = primsBoundingBoxMax;


	size_t everyThreadNum = SET_FLAG_WITH_SPLIT_THREAD_HANDLE_NUM;
	size_t threadNum	  = SET_FLAG_WITH_SPLIT_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	float  splitValue;
	size_t splitType;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < triNodeListSize)
		{
			splitType  = d_osplitTypeListAssist[addrIdx];
			splitValue = d_osplitListAssist[addrIdx];
			switch(splitType)
			{
			case 1:		// x轴上分割
				d_ileftChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[0][triIdx[addrIdx]].x <= splitValue) ? 1 : 0;
				d_irightChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[1][triIdx[addrIdx]].x >= splitValue) ? 1 : 0;
				break;
			case 2:		// y轴上分割
				d_ileftChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[0][triIdx[addrIdx]].y <= splitValue) ? 1 : 0;
				d_irightChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[1][triIdx[addrIdx]].y >= splitValue) ? 1 : 0;
				break;
			case 3:		// z轴上分割
				d_ileftChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[0][triIdx[addrIdx]].z <= splitValue) ? 1 : 0;
				d_irightChildListFlagsAssist[addrIdx] = 
					(primsBoundingBox[1][triIdx[addrIdx]].z >= splitValue) ? 1 : 0;
				break;
			default:
				break;
			}

		}
	}
}


// 对nextList的信息进行完善, 建立nextList与newTriNodeList之间的联系, 并对segment进行初始化
// 而且还对smallNode, largeNode的标志进行初始化
__global__
void finishNextListKernal(	int *nextList,
						  //~KdNodeOrigin    *kdNode,
						  KdNode_base       *kdNodeBase,
						  KdNode_bb         *kdNodeBB,
						  KdNode_extra      *kdNodeExtra,
						  size_t			kdSize,
						  size_t			nextListSize,
						  size_t          *nextNodeNumList,
						  size_t			*nextNodeAddr,
						  size_t			*newTriNodeList_segFlags,
						  size_t			*largeNodeFlagsList,
						  size_t			*largeNodeTriNodeFlagsList,
						  size_t			*smallNodeFlagsList,
						  size_t			*smallNodeTriNodeFlagsList)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	size_t everyThreadNum = FINISH_NEXT_LIST_THREAD_HANDLE_NUM;
	size_t threadNum	  = FINISH_NEXT_LIST_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	size_t addr;
	size_t num;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < nextListSize)
		{
			// 设置next-list中每个node的内容
			num = nextNodeNumList[addrIdx];
			addr = nextNodeAddr[addrIdx];

			// 6.3 xxx
			//nextList[addrIdx] = kdNode + kdSize + addrIdx; // nextList[addrIdx] = kdNode + kdSize; 2010 5.24 10.58
			//~kdNode[nextList[addrIdx]].nTris_ = num;
			//~kdNode[nextList[addrIdx]].addrTriNodeList_ = addr;
			int newNextList = nextList[addrIdx];
			kdNodeExtra[newNextList].nTris_ = num;
			kdNodeExtra[newNextList].addrTriNodeList_ = addr;

			// 设置newTriNodeList的segment标志
			newTriNodeList_segFlags[addr] = 1;

			// 对large/small node各种标志进行初始化
			if (num > LARGE_SMALL_SPLIT)
			{
				largeNodeFlagsList[addrIdx] = 1;
				largeNodeTriNodeFlagsList[addr] = 1;
			}
			else
			{
				smallNodeFlagsList[addrIdx] = 1;
				smallNodeTriNodeFlagsList[addr] = 1;
			}
		}
	}

}


// 对large/smallNodeNextList进行处理
__global__
void finishNodeNextListKernal(	int *nextList,
							  size_t			nextListSize,
							  size_t          *addrList,
							  size_t			*triNodeSegFlags,
							  //~KdNodeOrigin *kdNode,
							  KdNode_base      *kdNodeBase,
							  KdNode_bb        *kdNodeBB,
							  KdNode_extra     *kdNodeExtra,
							  size_t           relativeStart)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	size_t everyThreadNum = FILTER_FINISH_THREAD_HANDLE_NUM;
	size_t threadNum	  = FILTER_FINISH_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	int node;
	size_t addr;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < nextListSize)
		{
			// 重新设定triNode的起始地址，prims的个数相对不变
			node = nextList[addrIdx];
			addr = addrList[addrIdx];
			//~kdNode[node].addrTriNodeList_ = addr + relativeStart; // ***??????????
			kdNodeExtra[node].addrTriNodeList_ = addr + relativeStart;
			//kdNode->nTris_ =  ???

			// 设置segment flag
			triNodeSegFlags[addr] = 1;
		}
	}
}

__global__
void setSmallNodeRootMaskKernal(  size_t		*smallNodeRootMaskHi, 
								size_t		*smallNodeRootMaskLo,
								int *smallNodeList,
								size_t        *smallNodeBoundryFlags,
								//float         *boundryAddValue,
								size_t      *smallNodeSegStartAddr,
								size_t	    *smallNodeEveryLeafSize,
								size_t		*smallNodeRootList,
								size_t      *smallNodeRList,
								//~KdNodeOrigin *kdNode,
								KdNode_base      *kdNodeBase,
								KdNode_bb        *kdNodeBB,
								KdNode_extra     *kdNodeExtra,
								size_t		smallNodeListSize)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	size_t everyThreadNum = SET_SMALL_NODE_MASK_THREAD_HANDLE_NUM;
	size_t threadNum	  = SET_SMALL_NODE_MASK_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	size_t nodeSize;
	size_t segPos;
	size_t addrTri;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < smallNodeListSize)
		{
			//~nodeSize = kdNode[smallNodeList[addrIdx]].nTris_;
			int newSmallNode = smallNodeList[addrIdx];
			nodeSize = kdNodeExtra[newSmallNode].nTris_;
			
			// 为每个原始的small node设置位掩码
			smallNodeRootMaskHi[addrIdx] = d_rootMaskHi_[nodeSize - 1];
			smallNodeRootMaskLo[addrIdx] = d_rootMaskLo_[nodeSize - 1];

			// 原始small node索引
			smallNodeRList[addrIdx] = addrIdx;

			//~addrTri = kdNode[smallNodeList[addrIdx]].addrTriNodeList_;
			addrTri = kdNodeExtra[newSmallNode].addrTriNodeList_;
			segPos = addrTri * 2;


			// set boundry flags, 边界处设置1
			smallNodeBoundryFlags[segPos] = 1u;

			// 每个原始small node中tris的个数
			smallNodeEveryLeafSize[addrIdx] = nodeSize;

			smallNodeSegStartAddr[segPos] = segPos;// ???

			// 在每个segment边界处，设置其node的索引，便于以后的distribute
			smallNodeRootList[segPos] = addrIdx;
		}
	}
}

// 初始化boundry
__global__
void initBoundryKernal(//float4 **primsBoundingBox,
					   float4 *primsBoundingBoxMin,
					   float4 *primsBoundingBoxMax,

					   size_t *smallNodeTriIdx,
					   //float **smallNodeBoundryValue,
					   float  *smallNodeBoundryValueX,
					   float  *smallNodeBoundryValueY,
					   float  *smallNodeBoundryValueZ,

					   //size_t **smallNodeBoundryRPos,
					   size_t *smallNodeBoundryRPosX,
					   size_t *smallNodeBoundryRPosY,
					   size_t *smallNodeBoundryRPosZ,


					   //size_t **smallNodeBoundryType,
					   size_t *smallNodeBoundryTypeX,
					   size_t *smallNodeBoundryTypeY,
					   size_t *smallNodeBoundryTypeZ,

					   //size_t **smallNodeBoundryTriIdx,
					   size_t *smallNodeBoundryTriIdxX,
					   size_t *smallNodeBoundryTriIdxY,
					   size_t *smallNodeBoundryTriIdxZ,

					   //size_t **smallNodeBoundryAPos,
					   size_t *smallNodeBoundryAPosX,
					   size_t *smallNodeBoundryAPosY,
					   size_t *smallNodeBoundryAPosZ,

					   size_t *relativePos,
					   size_t *smallNodeSegStartAddr,
					   size_t smallNodeBoundrySize)
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;


	//
	float4 *primsBoundingBox[2];
	primsBoundingBox[0] = primsBoundingBoxMin;
	primsBoundingBox[1] = primsBoundingBoxMax;
	float  *smallNodeBoundryValue[3];
	smallNodeBoundryValue[0] = smallNodeBoundryValueX;
	smallNodeBoundryValue[1] = smallNodeBoundryValueY;
	smallNodeBoundryValue[2] = smallNodeBoundryValueZ;
	size_t *smallNodeBoundryRPos[3];
	smallNodeBoundryRPos[0] = smallNodeBoundryRPosX;
	smallNodeBoundryRPos[1] = smallNodeBoundryRPosY;
	smallNodeBoundryRPos[2] = smallNodeBoundryRPosZ;
	size_t *smallNodeBoundryType[3];
	smallNodeBoundryType[0] = smallNodeBoundryTypeX;
	smallNodeBoundryType[1] = smallNodeBoundryTypeY;
	smallNodeBoundryType[2] = smallNodeBoundryTypeZ;
	size_t *smallNodeBoundryTriIdx[3];
    smallNodeBoundryTriIdx[0] = smallNodeBoundryTriIdxX;
	smallNodeBoundryTriIdx[1] = smallNodeBoundryTriIdxY;
	smallNodeBoundryTriIdx[2] = smallNodeBoundryTriIdxZ;
	size_t *smallNodeBoundryAPos[3];
	smallNodeBoundryAPos[0] = smallNodeBoundryAPosX;
	smallNodeBoundryAPos[1] = smallNodeBoundryAPosY;
	smallNodeBoundryAPos[2] = smallNodeBoundryAPosZ;

	size_t everyThreadNum = INIT_BOUNDRY_THREAD_HANDLE_NUM;
	size_t threadNum	  = INIT_BOUNDRY_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	size_t triIdx;
	float4 bbMin;
	float4 bbMax;
	size_t halfAddrIdx;
	//float addValue;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < smallNodeBoundrySize)
		{
			// get triIdx and bb
			halfAddrIdx	= addrIdx >> 1;
			triIdx		= smallNodeTriIdx[halfAddrIdx];
			bbMin		= primsBoundingBox[0][triIdx];
			bbMax		= primsBoundingBox[1][triIdx];

			// boundry value,为每个split候选值
			// 左边界为最小值,右边界为最大值
			if (0 == (addrIdx & 1))
			{
				smallNodeBoundryValue[0][addrIdx] = bbMin.x;
				smallNodeBoundryValue[1][addrIdx] = bbMin.y;
				smallNodeBoundryValue[2][addrIdx] = bbMin.z;
			}
			else
			{
				smallNodeBoundryValue[0][addrIdx] = bbMax.x;
				smallNodeBoundryValue[1][addrIdx] = bbMax.y;
				smallNodeBoundryValue[2][addrIdx] = bbMax.z;
			}

			size_t segStartAddr = smallNodeSegStartAddr[addrIdx];
			for (int j = 0; j < 3; j++)
			{
				// relative pos
				smallNodeBoundryRPos[j][addrIdx]	= relativePos[halfAddrIdx];

				// triIdx
				smallNodeBoundryTriIdx[j][addrIdx]	= triIdx;

				// absolute pos????
				smallNodeBoundryAPos[j][addrIdx] = segStartAddr;

				if (0 == (addrIdx & 1))
				{
					// type start
					smallNodeBoundryType[j][addrIdx] = 1;
				}
				else
				{
					// type end
					smallNodeBoundryType[j][addrIdx] = 0;
				}

			}
		}
	}
}

__global__ 
void setLeftRightMaskKernal(	//size_t **smallNodeMaskLeftHi,	// split左边prims数目, 高位, 输出
							size_t *smallNodeMaskLeftHiX,
							size_t *smallNodeMaskLeftHiY,
							size_t *smallNodeMaskLeftHiZ,

							//size_t **smallNodeMaskLeftLo,	// split左边prims数目, 低位, 输出 
							size_t *smallNodeMaskLeftLoX,
							size_t *smallNodeMaskLeftLoY,
							size_t *smallNodeMaskLeftLoZ,

							//size_t **smallNodeMaskRightHi,	// split右边prims数目, 高位, 输出
							size_t *smallNodeMaskRightHiX,
							size_t *smallNodeMaskRightHiY,
							size_t *smallNodeMaskRightHiZ,

							//size_t **smallNodeMaskRightLo,	// split右边prims数目, 低位, 输出
							size_t *smallNodeMaskRightLoX,
							size_t *smallNodeMaskRightLoY,
							size_t *smallNodeMaskRightLoZ,

							//size_t **smallNodeBoundryRPos,		// 相对位置
							size_t *smallNodeBoundryRPosX,
							size_t *smallNodeBoundryRPosY,
							size_t *smallNodeBoundryRPosZ,


							//size_t **smallNodeBoundryAPos,		// 绝对位置
							size_t *smallNodeBoundryAPosX,
							size_t *smallNodeBoundryAPosY,
							size_t *smallNodeBoundryAPosZ,

							//size_t **smallNodeBoundryType,		// 类型
							size_t *smallNodeBoundryTypeX,
							size_t *smallNodeBoundryTypeY,
							size_t *smallNodeBoundryTypeZ,
							
							size_t *smallNodeRootList,
							size_t *smallNodeRootMaskHi,
							size_t *smallNodeRootMaskLo,
							size_t smallNodeBoundrySize
							
							//// 
							//int *smallNodeList,
							//KdNodeOrigin *kdNode
							)	

{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	//
    __shared__ size_t *smallNodeBoundryType[4];
	__shared__ size_t *smallNodeBoundryAPos[4];
	__shared__ size_t *smallNodeBoundryRPos[4];
	__shared__ size_t *smallNodeMaskLeftHi[4];
	__shared__ size_t *smallNodeMaskLeftLo[4];
	__shared__ size_t *smallNodeMaskRightHi[4];
	__shared__ size_t *smallNodeMaskRightLo[4];

	if (threadIdx.x == 0){
	smallNodeBoundryType[0] = smallNodeBoundryTypeX;
	smallNodeBoundryType[1] = smallNodeBoundryTypeY;
	smallNodeBoundryType[2] = smallNodeBoundryTypeZ;
	
	smallNodeBoundryAPos[0] = smallNodeBoundryAPosX;
	smallNodeBoundryAPos[1] = smallNodeBoundryAPosY;
	smallNodeBoundryAPos[2] = smallNodeBoundryAPosZ;
	
	smallNodeBoundryRPos[0] = smallNodeBoundryRPosX;
	smallNodeBoundryRPos[1] = smallNodeBoundryRPosY;
	smallNodeBoundryRPos[2] = smallNodeBoundryRPosZ;
    
	smallNodeMaskLeftHi[0] = smallNodeMaskLeftHiX;
	smallNodeMaskLeftHi[1] = smallNodeMaskLeftHiY;
	smallNodeMaskLeftHi[2] = smallNodeMaskLeftHiZ;
	
	smallNodeMaskLeftLo[0] = smallNodeMaskLeftLoX;
	smallNodeMaskLeftLo[1] = smallNodeMaskLeftLoY;
	smallNodeMaskLeftLo[2] = smallNodeMaskLeftLoZ;
	
	smallNodeMaskRightHi[0] = smallNodeMaskRightHiX;
	smallNodeMaskRightHi[1] = smallNodeMaskRightHiY;
	smallNodeMaskRightHi[2] = smallNodeMaskRightHiZ;
	
	smallNodeMaskRightLo[0] = smallNodeMaskRightLoX;
	smallNodeMaskRightLo[1] = smallNodeMaskRightLoY;
	smallNodeMaskRightLo[2] = smallNodeMaskRightLoZ;
	}
	__syncthreads();








	size_t everyThreadNum = SET_LEFT_RIGHT_MASK_THREAD_HANDLE_NUM;
	size_t threadNum	  = SET_LEFT_RIGHT_MASK_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	size_t aPos;
	size_t rPos;
	size_t type;
	size_t nodeIdx;
	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < smallNodeBoundrySize)
		{
			nodeIdx = smallNodeRootList[addrIdx];
			for (size_t j = 0; j < 3; j++)
			{
				aPos = smallNodeBoundryAPos[j][addrIdx];
				rPos = smallNodeBoundryRPos[j][aPos];
				type = smallNodeBoundryType[j][aPos];
				
				//int num = kdNode[smallNodeList[nodeIdx]].nTris_;

				if (1 == type)	// start
				{
					// left
					smallNodeMaskLeftHi[j][addrIdx] = d_rootStartMaskHi_[rPos - 1];
					smallNodeMaskLeftLo[j][addrIdx] = d_rootStartMaskLo_[rPos - 1];

					// right
					smallNodeMaskRightHi[j][addrIdx] = ((unsigned int)(-1));
					smallNodeMaskRightLo[j][addrIdx] = ((unsigned int)(-1));
					smallNodeMaskRightHi[j][addrIdx] &=  smallNodeRootMaskHi[nodeIdx];
					smallNodeMaskRightLo[j][addrIdx] &=  smallNodeRootMaskLo[nodeIdx];

				}
				else			// end
				{
					// left
					smallNodeMaskLeftHi[j][addrIdx] = 0u;
					smallNodeMaskLeftLo[j][addrIdx] = 0u;

					// right
					smallNodeMaskRightHi[j][addrIdx] = d_rootEndMaskHi_[rPos - 1];
					smallNodeMaskRightLo[j][addrIdx] = d_rootEndMaskLo_[rPos - 1];
					smallNodeMaskRightHi[j][addrIdx] &=  smallNodeRootMaskHi[nodeIdx];
					smallNodeMaskRightLo[j][addrIdx] &=  smallNodeRootMaskLo[nodeIdx];
				} 
			}
		}
	}
}

// process small node kernal
__global__
void processSmallNodeKernal(size_t *smallNodeMaskHi,					// mask
							size_t *smallNodeMaskLo,

							size_t *smallNodeRList,						// root

							size_t *smallNodeRootMaskHi,				// root mask
							size_t *smallNodeRootMaskLo,

							//size_t **smallNodeMaskLeftHi,				// root split mask
							size_t *smallNodeMaskLeftHiX,
							size_t *smallNodeMaskLeftHiY,
							size_t *smallNodeMaskLeftHiZ,

							//size_t **smallNodeMaskLeftLo,
							size_t *smallNodeMaskLeftLoX,
							size_t *smallNodeMaskLeftLoY,
							size_t *smallNodeMaskLeftLoZ,

							//size_t **smallNodeMaskRightHi,
							size_t *smallNodeMaskRightHiX,
							size_t *smallNodeMaskRightHiY,
							size_t *smallNodeMaskRightHiZ,

							//size_t **smallNodeMaskRightLo,
							size_t *smallNodeMaskRightLoX,
							size_t *smallNodeMaskRightLoY,
							size_t *smallNodeMaskRightLoZ,


							//size_t **smallNodeBoundryAPos,				// absolute position
							size_t *smallNodeBoundryAPosX,
							size_t *smallNodeBoundryAPosY,
							size_t *smallNodeBoundryAPosZ,

							//size_t **smallNodeBoundryRPos,
							size_t *smallNodeBoundryRPosX,
							size_t *smallNodeBoundryRPosY,
							size_t *smallNodeBoundryRPosZ,


							int *smallNodeRoot,
							int *smallNodeList,				// small node list

							size_t *smallNodeLeafFlags,					// leaf flags
							size_t *smallNodeNoLeafFlags,				// nonleaf flags
							size_t *smallNodeLeafSize,					// leaf prims size

							//float  **smallNodeBoundryValue,
							float *smallNodeBoundryValueX,
							float *smallNodeBoundryValueY,
							float *smallNodeBoundryValueZ,

							//KdNodeOrigin *kdNode,
							KdNode_base* kdNodeBase,
							KdNode_bb*   kdNodeBB,
							KdNode_extra* kdNodeExtra,


							float tCost,								// traversal cost
							float iCost,								// intersection cost
							
							size_t smallProcTime,
							size_t smallNodeListSize
							)
{
	//size_t tidx = threadIdx.x;
	//size_t bidx = blockIdx.x;
	size_t addrIdx	= blockIdx.x * blockDim.x + threadIdx.x;

	// 
    __shared__ float *smallNodeBoundryValue[4];
	__shared__ size_t *smallNodeBoundryRPos[4];
	__shared__ size_t *smallNodeBoundryAPos[4];
	__shared__ size_t *smallNodeMaskLeftHi[4];
	__shared__ size_t *smallNodeMaskLeftLo[4];
	__shared__ size_t *smallNodeMaskRightHi[4];
	__shared__ size_t *smallNodeMaskRightLo[4];

	if (threadIdx.x == 0){
	smallNodeBoundryValue[0] = smallNodeBoundryValueX;
	smallNodeBoundryValue[1] = smallNodeBoundryValueY;
	smallNodeBoundryValue[2] = smallNodeBoundryValueZ;
	
	smallNodeBoundryRPos[0] = smallNodeBoundryRPosX;
	smallNodeBoundryRPos[1] = smallNodeBoundryRPosY;
	smallNodeBoundryRPos[2] = smallNodeBoundryRPosZ;
	
	smallNodeBoundryAPos[0] = smallNodeBoundryAPosX;
	smallNodeBoundryAPos[1] = smallNodeBoundryAPosY;
	smallNodeBoundryAPos[2] = smallNodeBoundryAPosZ;
	
	smallNodeMaskLeftHi[0] = smallNodeMaskLeftHiX;
	smallNodeMaskLeftHi[1] = smallNodeMaskLeftHiY;
	smallNodeMaskLeftHi[2] = smallNodeMaskLeftHiZ;
	
	smallNodeMaskLeftLo[0] = smallNodeMaskLeftLoX;
	smallNodeMaskLeftLo[1] = smallNodeMaskLeftLoY;
	smallNodeMaskLeftLo[2] = smallNodeMaskLeftLoZ;
	
	smallNodeMaskRightHi[0] = smallNodeMaskRightHiX;
	smallNodeMaskRightHi[1] = smallNodeMaskRightHiY;
	smallNodeMaskRightHi[2] = smallNodeMaskRightHiZ;
	
	smallNodeMaskRightLo[0] = smallNodeMaskRightLoX;
	smallNodeMaskRightLo[1] = smallNodeMaskRightLoY;
	smallNodeMaskRightLo[2] = smallNodeMaskRightLoZ;
	}

	__syncthreads();

	//__shared__ float4 bbMaxs[PROCESS_SMALL_NODE_BLOCK_SIZE];
	__shared__ BoundingBox bbs[PROCESS_SMALL_NODE_BLOCK_SIZE];
	__shared__ FLOAT4 ranges[PROCESS_SMALL_NODE_BLOCK_SIZE];
 


	size_t rootIdx;
	size_t maskHi, maskLo;
	size_t rootMaskHi, rootMaskLo;
	size_t maskHiTemp, maskLoTemp;

	//BoundingBox bb;
	//float  bbMin[3], bbMax[3];
	//float  range[3];
	size_t axisType, otherAxis0, otherAxis1;

	int bestAxis = -1;
	float  bestSplit = FLOAT_MIN;
	float  bestCost = FLOAT_MAX;
	float  oldCost, cost;

	size_t bestRPos;

	size_t rootNPrims, nPrims;

	size_t splitStart;

	float  totalSA;
	float  invTotalSA;
	float  belowSA, aboveSA;
	size_t nBelow, nAbove;
	float  pBelow, pAbove;

	float  splitValue;


	size_t aPos;
	size_t rPos;

	for (size_t i = 0; i < PROCESS_SMALL_NODE_THREAD_HANDLE_NUM; i++)
	{
		addrIdx = i * (PROCESS_SMALL_NODE_BLOCK_SIZE * gridDim.x) + addrIdx;
		if (addrIdx < smallNodeListSize)
		{
			//
			bestAxis = -1;
			bestSplit = FLOAT_MIN;
			bestCost = FLOAT_MAX;

			rootIdx = smallNodeRList[addrIdx];
			rootMaskHi = smallNodeRootMaskHi[rootIdx];
			rootMaskLo = smallNodeRootMaskLo[rootIdx];
			maskHi = smallNodeMaskHi[addrIdx];
			maskLo = smallNodeMaskLo[addrIdx];


			// 根节点prims数目
			rootNPrims = fastBitCounting64(rootMaskHi, rootMaskLo);

			// 当前节点个数
			nPrims = fastBitCounting64(maskHi, maskLo);

			if (
				nPrims < LEAF_THRESHOLD || smallProcTime >= SMALL_HANDLE_MAX)	
			{
				kdNodeBase[smallNodeList[addrIdx]].splitType_ = 4;
				kdNodeExtra[smallNodeList[addrIdx]].nTris_ = nPrims;
				kdNodeBase[smallNodeList[addrIdx]].lChild_ = -1;
				kdNodeBase[smallNodeList[addrIdx]].rChild_ = -1;


				// seg flag
				smallNodeLeafFlags[addrIdx] = 1;
				smallNodeNoLeafFlags[addrIdx] = 0;
				smallNodeLeafSize[addrIdx] = nPrims;
				return;
			}

			// leaf cost
			oldCost = nPrims * iCost;

			// 决定x, y, z轴split
			/*bb = kdNodeBB[smallNodeList[addrIdx]].bb_;
			bbMin[0] = bb.bbMin_.x;
			bbMin[1] = bb.bbMin_.y;
			bbMin[2] = bb.bbMin_.z;
			bbMax[0] = bb.bbMax_.x;
			bbMax[1] = bb.bbMax_.y;
			bbMax[2] = bb.bbMax_.z;

			range[0] = bbMax[0] - bbMin[0];
			range[1] = bbMax[1] - bbMin[1];
			range[2] = bbMax[2] - bbMin[2];*/
			bbs[threadIdx.x] = kdNodeBB[smallNodeList[addrIdx]].bb_;
			ranges[threadIdx.x].x = bbs[threadIdx.x].bbMax_.x - bbs[threadIdx.x].bbMin_.x;
			ranges[threadIdx.x].y = bbs[threadIdx.x].bbMax_.y - bbs[threadIdx.x].bbMin_.y;
			ranges[threadIdx.x].z = bbs[threadIdx.x].bbMax_.z - bbs[threadIdx.x].bbMin_.z;
			__syncthreads();


			// axisType = (range[0] > range[1]) ? ((range[0] > range[2])? 0 : 2) : ((range[1] > range[2])? 1 : 2); 
			axisType =	(ranges[threadIdx.x].x > ranges[threadIdx.x].y) ? 
						((ranges[threadIdx.x].x > ranges[threadIdx.x].z)? 0 : 2) :
						((ranges[threadIdx.x].y > ranges[threadIdx.x].z)? 1 : 2); 


			int retries = 0;

			while(1)
			{
				otherAxis0 = (axisType + 1) % 3;
				otherAxis1 = (axisType + 2) % 3;

				// 整个node的面积
				//totalSA = (range[0] * range[1] + range[0] * range[2] + range[1] * range[2]) * 2;
				totalSA = (	ranges[threadIdx.x].x * ranges[threadIdx.x].y + 
							ranges[threadIdx.x].x * ranges[threadIdx.x].z + 
							ranges[threadIdx.x].y * ranges[threadIdx.x].z) * 2;
				invTotalSA = 1.0f / totalSA;

				// 遍历所有的split
				splitStart = kdNodeExtra[smallNodeRoot[rootIdx]].addrTriNodeList_ << 1;
				for (size_t k = 0; k < 2 * rootNPrims; k++)
				{
					// 绝对位置
					aPos = smallNodeBoundryAPos[axisType][splitStart+k];
					rPos = smallNodeBoundryRPos[axisType][aPos];

					// 判断当前split所属的三角形是否包含在当前节点三角形集合中
					if (0 == (d_rootStartMaskHi_[rPos - 1] & maskHi) && 
						0 == (d_rootStartMaskLo_[rPos - 1] & maskLo))
					{
					    continue;
					}
					

					// 当前的split值
					//splitValue = smallNodeBoundryValue[axisType][aPos];
					splitValue = smallNodeBoundryValue[axisType][splitStart+k];

					// 当前左右孩子节点的数目
					maskHiTemp = smallNodeMaskLeftHi[axisType][splitStart + k];
					maskLoTemp = smallNodeMaskLeftLo[axisType][splitStart + k];
					nBelow = fastBitCounting64(maskHi & maskHiTemp, 
						maskLo & maskLoTemp);

					maskHiTemp = smallNodeMaskRightHi[axisType][splitStart + k];
					maskLoTemp = smallNodeMaskRightLo[axisType][splitStart + k];
					nAbove = fastBitCounting64(maskHi & maskHiTemp,
						maskLo & maskLoTemp);

					// split值在包围盒之内，才能考虑
					/*if (splitValue > bbMin[axisType] && 
						splitValue < bbMax[axisType])*/
					if (splitValue > bbs[threadIdx.x].bbMin_.f[axisType] && 
						splitValue < bbs[threadIdx.x].bbMax_.f[axisType])
					{
						/*belowSA = 2 * (range[otherAxis0] * range[otherAxis1] + 
							(splitValue - bbMin[axisType]) * (range[otherAxis0] + range[otherAxis1])); 
						aboveSA = 2 * (range[otherAxis0] * range[otherAxis1] + 
							(bbMax[axisType] - splitValue) * (range[otherAxis0] + range[otherAxis1])) ;*/
						belowSA = 2 * (ranges[threadIdx.x].f[otherAxis0] * ranges[threadIdx.x].f[otherAxis1] + 
							(splitValue - bbs[threadIdx.x].bbMin_.f[axisType]) * (ranges[threadIdx.x].f[otherAxis0] + ranges[threadIdx.x].f[otherAxis1])); 
						aboveSA = 2 * (ranges[threadIdx.x].f[otherAxis0] * ranges[threadIdx.x].f[otherAxis1] + 
							(bbs[threadIdx.x].bbMax_.f[axisType]- splitValue) * (ranges[threadIdx.x].f[otherAxis0] + ranges[threadIdx.x].f[otherAxis1])); 
						pBelow = belowSA * invTotalSA;
						pAbove = aboveSA * invTotalSA;

						// cost 
						cost = tCost + iCost * (pBelow * nBelow + pAbove * nAbove);

						// 更新最好的split
						if (cost < bestCost)
						{
							bestCost = cost;
							bestAxis = axisType;
							bestSplit = splitValue;

							bestRPos = splitStart + k;
						}					
					}
				}

				// 在下一个轴上选split
				if (bestAxis == -1 && retries < 2)
				{
					retries ++;
					axisType = (axisType + 1) % 3;
				}
				else
				{
					break;
				}
			}
			// decide a node is leaf or continue split
			if (bestAxis == -1 || 
				bestCost > oldCost || 
				nPrims < LEAF_THRESHOLD || smallProcTime >= SMALL_HANDLE_MAX)	
			{
				kdNodeBase[smallNodeList[addrIdx]].splitType_ = 4;
				kdNodeExtra[smallNodeList[addrIdx]].nTris_ = nPrims;
				kdNodeBase[smallNodeList[addrIdx]].lChild_ = -1;
				kdNodeBase[smallNodeList[addrIdx]].rChild_ = -1;


				// seg flag
				smallNodeLeafFlags[addrIdx] = 1;
				smallNodeNoLeafFlags[addrIdx] = 0;
				smallNodeLeafSize[addrIdx] = nPrims;
			}
			else
			{ 
				// seg flag
				smallNodeLeafFlags[addrIdx] = 0;
				smallNodeNoLeafFlags[addrIdx] = 1;

				kdNodeBase[smallNodeList[addrIdx]].splitType_ = bestAxis + 1;
				kdNodeBase[smallNodeList[addrIdx]].splitValue_ = bestSplit;

				// set mask
				kdNodeBase[smallNodeList[addrIdx]].rChild_ = bestRPos;

				
				
			}


		}
	}
}

// 整理叶子节点
__global__
void leafNodeFilterKernal(	
						  size_t  *smallNodeNoLeafFlags,	// non leaf
						  size_t  *smallNodeNoLeafAddr,

						  size_t  *smallNodeLeafAddr,		// leaf 
						  size_t  *smallNodeLeafPrimsSize,
						  size_t  *smallNodeLeafPrimsAddr,

						  size_t  *smallNodeLeafPrims,
						  size_t  leafSize,


						  int *smallNodeRoot,
						  int *smallNodeList,			// node list
						  int *smallNodeNextList,		// next node list

						  size_t *smallNodeRList,				// root list		
						  size_t *smallNodeNextListRList,		// next root list



						  size_t *smallNodeMaskHi, 
						  size_t *smallNodeMaskLo,
						  size_t *smallNodeNextListMaskHi, 
						  size_t *smallNodeNextListMaskLo,

						  size_t *triIdx,

						  //~KdNodeOrigin *kdNode,
						  KdNode_base *kdNodeBase,
						  KdNode_bb *kdNodeBB,
						  KdNode_extra *kdNodeExtra,
						  size_t        kdSize,

						  //size_t **smallNodeMaskLeftHi,
						  size_t *smallNodeMaskLeftHiX,
						  size_t *smallNodeMaskLeftHiY,
						  size_t *smallNodeMaskLeftHiZ,

						  //size_t **smallNodeMaskLeftLo,
						  size_t *smallNodeMaskLeftLoX,
						  size_t *smallNodeMaskLeftLoY,
						  size_t *smallNodeMaskLeftLoZ,

						  //size_t **smallNodeMaskRightHi,
						  size_t *smallNodeMaskRightHiX,
						  size_t *smallNodeMaskRightHiY,
						  size_t *smallNodeMaskRightHiZ,

						  //size_t **smallNodeMaskRightLo,
						  size_t *smallNodeMaskRightLoX,
						  size_t *smallNodeMaskRightLoY,
						  size_t *smallNodeMaskRightLoZ,


						  size_t smallNodeListSize

						  // test
						  //size_t *smallCpy,
						  //size_t *result
						  )
{
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx	= bidx * blockDim.x + tidx;

	//
    __shared__ size_t *smallNodeMaskLeftHi[4];
	__shared__ size_t *smallNodeMaskLeftLo[4];
	__shared__ size_t *smallNodeMaskRightHi[4];
	__shared__ size_t *smallNodeMaskRightLo[4];

	if (threadIdx.x == 0){
	smallNodeMaskLeftHi[0] = smallNodeMaskLeftHiX;
	smallNodeMaskLeftHi[1] = smallNodeMaskLeftHiY;
	smallNodeMaskLeftHi[2] = smallNodeMaskLeftHiZ;
	
	smallNodeMaskLeftLo[0] = smallNodeMaskLeftLoX;
	smallNodeMaskLeftLo[1] = smallNodeMaskLeftLoY;
	smallNodeMaskLeftLo[2] = smallNodeMaskLeftLoZ;
	
	smallNodeMaskRightHi[0] = smallNodeMaskRightHiX;
	smallNodeMaskRightHi[1] = smallNodeMaskRightHiY;
	smallNodeMaskRightHi[2] = smallNodeMaskRightHiZ;
	
	smallNodeMaskRightLo[0] = smallNodeMaskRightLoX;
	smallNodeMaskRightLo[1] = smallNodeMaskRightLoY;
	smallNodeMaskRightLo[2] = smallNodeMaskRightLoZ;
	}
	__syncthreads();



	size_t everyThreadNum = LEAF_FILTER_THREAD_HANDLE_NUM;
	size_t threadNum	  = LEAF_FILTER_BLOCK_SIZE * gridDim.x;
	size_t addrIdx;

	for (size_t i = 0; i < everyThreadNum; i++)
	{
		addrIdx = i * threadNum + idx;
		if (addrIdx < smallNodeListSize)
		{
			
			if (1  == smallNodeNoLeafFlags[addrIdx]) // 处理非叶子节点
			{
				// 得到split的左右mask
				/*size_t bestMaskLeftHi = smallNodeList[addrIdx]->addrTriNodeList_;
				size_t bestMaskLeftLo = smallNodeList[addrIdx]->nTris_;
				size_t bestMaskRightHi = (size_t)smallNodeList[addrIdx]->lChild_;
				size_t bestMaskRightLo = (size_t)smallNodeList[addrIdx]->rChild_;*/
				//~size_t splitType = kdNode[smallNodeList[addrIdx]].splitType_;
				//~float  splitValue = kdNode[smallNodeList[addrIdx]].splitValue_;
				int newSmallNode = smallNodeList[addrIdx];
				KdNode_base newKdNodeBase = kdNodeBase[newSmallNode];
				size_t splitType = newKdNodeBase.splitType_;
				float  splitValue = newKdNodeBase.splitValue_;

				//~size_t bestRPos = (size_t)kdNode[smallNodeList[addrIdx]].rChild_;
				size_t bestRPos = (size_t)kdNodeBase[newSmallNode].rChild_;
				
				size_t bestMaskLeftHi = smallNodeMaskLeftHi[splitType - 1][bestRPos];
				size_t bestMaskLeftLo = smallNodeMaskLeftLo[splitType - 1][bestRPos];
				size_t bestMaskRightHi = smallNodeMaskRightHi[splitType - 1][bestRPos];
				size_t bestMaskRightLo = smallNodeMaskRightLo[splitType - 1][bestRPos];


				// 得到next的地址
				size_t nextAddr = (smallNodeNoLeafAddr[addrIdx] - 1) * 2;

				// next list
				smallNodeNextList[nextAddr] = kdSize + nextAddr;
				smallNodeNextList[nextAddr + 1] = kdSize + nextAddr + 1;

				//~kdNode[smallNodeList[addrIdx]].lChild_  = kdSize + nextAddr;
				//~kdNode[smallNodeList[addrIdx]].rChild_ = kdSize + nextAddr + 1;
				kdNodeBase[newSmallNode].lChild_ = kdSize + nextAddr;
				kdNodeBase[newSmallNode].rChild_ = kdSize + nextAddr + 1;

				// set child 
				BoundingBox leftBB, rightBB;
				//~leftBB = rightBB = kdNode[smallNodeList[addrIdx]].bb_;
				leftBB = rightBB = kdNodeBB[newSmallNode].bb_;
				
				switch(splitType)
				{
				case 1:
					leftBB.bbMax_.x = splitValue;
					rightBB.bbMin_.x = splitValue;
					break;
				case 2:
					leftBB.bbMax_.y = splitValue;
					rightBB.bbMin_.y = splitValue;
					break;
				case 3:
					leftBB.bbMax_.z = splitValue;
					rightBB.bbMin_.z = splitValue;
					break;
				default:
					break;
				}
				// child's bb
				//~kdNode[kdNode[smallNodeList[addrIdx]].lChild_].bb_ = leftBB;
				//~kdNode[kdNode[smallNodeList[addrIdx]].rChild_].bb_ = rightBB;
				newKdNodeBase = kdNodeBase[newSmallNode];
				kdNodeBB[newKdNodeBase.lChild_].bb_ = leftBB;
				kdNodeBB[newKdNodeBase.rChild_].bb_ = rightBB;

				// root
				smallNodeNextListRList[nextAddr] = smallNodeRList[addrIdx];
				smallNodeNextListRList[nextAddr + 1] = smallNodeRList[addrIdx];

				// mask
				size_t maskLeftHi = smallNodeMaskHi[addrIdx] & bestMaskLeftHi;
				size_t maskLeftLo = smallNodeMaskLo[addrIdx] & bestMaskLeftLo;
				size_t maskRightHi =  smallNodeMaskHi[addrIdx] & bestMaskRightHi;
				size_t maskRightLo = smallNodeMaskLo[addrIdx] & bestMaskRightLo;

				smallNodeNextListMaskHi[nextAddr] = maskLeftHi;
				smallNodeNextListMaskLo[nextAddr] = maskLeftLo;

				smallNodeNextListMaskHi[nextAddr + 1] = maskRightHi;
				smallNodeNextListMaskLo[nextAddr + 1] = maskRightLo;

				// child' nTris
				//~kdNode[kdNode[smallNodeList[addrIdx]].lChild_].nTris_ = fastBitCounting64(maskLeftHi, maskLeftLo);
				//~kdNode[kdNode[smallNodeList[addrIdx]].rChild_].nTris_ = fastBitCounting64(maskRightHi, maskRightLo);
				kdNodeExtra[newKdNodeBase.lChild_].nTris_ = fastBitCounting64(maskLeftHi, maskLeftLo);
				kdNodeExtra[newKdNodeBase.rChild_].nTris_ = fastBitCounting64(maskRightHi, maskRightLo);

			}
			else									 // 处理叶子节点
			{
				// size & addr
				size_t addr      = smallNodeLeafAddr[addrIdx] - 1;
				//size_t size      = smallNodeLeafPrimsSize[addr];
				size_t primsAddr = smallNodeLeafPrimsAddr[addr];

				// 设置kd node
				size_t* iaddr;
				//~iaddr = (size_t *)&kdNode[smallNodeList[addrIdx]].splitValue_;
				int newSmallNode = smallNodeList[addrIdx];
				iaddr = (size_t *)&kdNodeBase[newSmallNode].splitValue_;
				*iaddr = leafSize + primsAddr;
			   // kdNode[smallNodeList[addrIdx]].splitValue_ = (float)leafSize + primsAddr;
				

				// mask
				size_t maskHi = smallNodeMaskHi[addrIdx];
				size_t maskLo = smallNodeMaskLo[addrIdx];

				// root 
				size_t rootIdx = smallNodeRList[addrIdx];
				//~size_t rootStart = kdNode[smallNodeRoot[rootIdx]].addrTriNodeList_;
				size_t rootStart = kdNodeExtra[smallNodeRoot[rootIdx]].addrTriNodeList_;
				size_t rootAddr;

				size_t triAddr;

				// 收集叶子节点当中的prims

				size_t ones;
				size_t shifts;
				size_t mask;
				size_t oneAddr;
				size_t segMask;

				size_t triCount = 0;

				// 低位
				triAddr = leafSize + primsAddr;
				for (size_t i = 0; i < 4; i++)
				{
					shifts = i * 8;
					mask = (maskLo >> shifts);
					segMask = mask & 0xffu;
					ones = d_bits_in_char[segMask];
					for (size_t j = 0; j < ones; j++)
					{
						oneAddr = 64 - 8 * (i + 1) + d_one_loc[segMask * 8 + j]; // d_ones_loc[mask][j]
						rootAddr = rootStart + oneAddr;

						smallNodeLeafPrims[triAddr] = triIdx[rootAddr];
						triAddr ++;
						triCount ++;
					}
				}

				// 高位
				for (size_t i = 0; i < 4; i++)
				{
					shifts = i * 8;
					mask = (maskHi >> shifts);
					segMask = mask & 0xffu;
					ones = d_bits_in_char[segMask];
					for (size_t j = 0; j < ones; j++)
					{
						oneAddr = 32 - 8 * (i + 1) + d_one_loc[segMask * 8 + j];
						rootAddr = rootStart + oneAddr;

						smallNodeLeafPrims[triAddr] = triIdx[rootAddr];
						triAddr ++;
						triCount ++;
					}
				}

				//~kdNode[smallNodeList[addrIdx]].nTris_ = triCount;
				kdNodeExtra[newSmallNode].nTris_ = triCount;
			}	
		}
	}

}

// partition
__device__
inline int pivot(int r, 
				 int q, 
				 float *sm_data, 
				 unsigned char *sm_pos)
{
    int i = r - 1;
	float xKey;
	unsigned char xValue;
	float tempKey;
	unsigned char tempValue;
	int j;

	// select mid as pivot
	int pos = (r + q) / 2 ;
	tempKey = sm_data[q];
	sm_data[q] = sm_data[pos];
	sm_data[pos] = tempKey;

	tempValue = sm_pos[q];
	sm_pos[q] = sm_pos[pos];
	sm_pos[pos] = tempValue;

	xKey = sm_data[q];
	xValue = sm_pos[q];

	for(j = r; j < q; j++)
	{
	    if(sm_data[j] <= xKey)
		{
		    tempKey = sm_data[j];
			sm_data[j] = sm_data[++i];
			sm_data[i] = tempKey;

			i--;
			tempValue = sm_pos[j];
			i++;
			sm_pos[j] = sm_pos[i];
			sm_pos[i] = tempValue;
			
		}
	}

	tempKey = xKey;
	sm_data[q] = sm_data[i+1];
	sm_data[i+1] = xKey;

	tempValue = xValue;
	sm_pos[q] = sm_pos[i+1];
	sm_pos[i+1] = xValue;

	return i + 1;
}

// 非递归的快速排序
__device__
inline void quickSort(int lo, 
					  int hi, 
					  float *sm_data, 
					  uchar2 *sm_stack, 
					  unsigned char *sm_pos)
{
	//size_t base = 0;
	size_t top = 0;
	size_t size = 0;
	size_t base = lo;

	//size_t originR = lo;
	while(1)
	{
	    while(lo < hi)
		{
			int p = pivot(lo, hi,  sm_data, sm_pos);
		
			uchar2 rightPos;
			rightPos.x = p + 1 - base;
			rightPos.y = hi - base;
			sm_stack[top++] = rightPos;
			size++;
			
			hi = p - 1;
		}
		if (size <= 0) break;
		uchar2 newPos = sm_stack[--top];
		size--;
		lo = newPos.x + base;
		hi = newPos.y + base;
	}
    
}  

// 算法描述：
// 该算法是并行的segment快速排序算法, 每个block有16个线程, 每个线程负责最多128 = 64 * 2个数排序
// 上面的block设计是因为每个block中最多有16K的可供使用
// 为了提高排序的速度，要用到share memory, 因为数据之间打交道的次数比较频繁
// share memory是按如下方式进行布局：
// 第1块：用来存储待排序的数据
// 第2块：用来存储进行非递归的快速排序中使用的栈(CUDA不支持递归调用)
// 第3块: 用来存储待排序的索引
/* |<---------------------------------128 * 4 * 16-------------------------------->|
	 * |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  // data share mem
	 *
	 * <--------------128 * 2 * 16--------------------->
	 * |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |   // stack
	 *
	 * <--------------128 * 1 * 16----->
	 * | | | | | | | | | | | | | | | | |		// pos
	 */
__global__
void segParallelQuickSort(float *segData,  
				  size_t *segPos, 
				  size_t *numList, 
				  size_t *addrList, 
				  size_t nSeg)
{
    size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;
	size_t blockStart = bid * blockDim.x;	// 当前block开始的位置(相对于全局)
	size_t id = blockStart + tid;			// 当前线程处理数据的开始位置(相对于全局)

	//if (id < nSeg){

	// share memory
	extern __shared__ unsigned char sm[];
	
	// 当前线程在当前block中开始处理数据位置(相对于当前block)
	size_t sm_dataEveryThreadStartIdx;
	if (id < nSeg)
	{
	    sm_dataEveryThreadStartIdx = (addrList[id] - addrList[blockStart]) * 2;
	}

	// 当前线程在快速排序中用到的栈在share memory的开始地址
	size_t sm_stackIdx = 8192 + tid * 2 * 128;
	uchar2 *sm_stack = (uchar2 *)(sm + sm_stackIdx);

	// 当前block存储数据在share memory的开始地址
	float *sm_dataBegin = (float *)sm;

	// 当前block存储位置在share memory的开始地址
	unsigned char *sm_posBegin = sm + 12288;

	// 当前block要进行处理的数据开始的位置(相对于全局)
	size_t everyBlockStartDataIdx = addrList[blockStart] * 2;

	// 当前block处理数据的数目
	size_t blockHandleNum;
	if ((blockStart + blockDim.x) <= nSeg)
	{
	    blockHandleNum = (addrList[blockStart + 15] + 
						 numList[blockStart + 15] -
						 addrList[blockStart]) * 2;
		
	}
	else
	{
	    blockHandleNum = (addrList[nSeg - 1] + 
						 numList[nSeg - 1] - 
						 addrList[blockStart]) * 2;
	}

	// 当前block中每个线程读取数据的次数
	// 这里读取数据是整个block中的线程连续读取，这样可以最大程度的做到transaction
	// 传统方式: |...t1...|...t2...|...t3...|
	// 改进方式: |t1t2t3|t1t2t3|t1t2t3|t1t2t3|t1t2t3|t1t2t3|
	size_t readTime = (blockHandleNum & (blockDim.x - 1)) ? blockHandleNum / 
					   blockDim.x + 1 : blockHandleNum / blockDim.x;
	size_t addrIdx;
	
	// 将待排序的数据从global memory装载到share memory
	for (int i = 0; i < readTime; i++)
	{
	    addrIdx = i * blockDim.x + tid;
		if (addrIdx < blockHandleNum)
		{
		    sm_dataBegin[addrIdx] = segData[everyBlockStartDataIdx + addrIdx];
		}
	}

	__syncthreads();

	// 设置当前线程所处理的数据的相对位置, 这里除了对数据进行排序外，数据的相对位置也
	// 同时进行了排序，得到相对位置的排序，目的是要知道排在第i位的是哪一个数据
	// 0 1 2 3 4 5 6 -------
	size_t num;
	if (id < nSeg)
	{
	    num = numList[id] * 2;
		for (int i = 0; i < num; i++)
		{
		    sm_posBegin[sm_dataEveryThreadStartIdx + i] = i;
		}
	}

	__syncthreads();


	// 当前线程进行快速排序
	if (id < nSeg)
	{
	    size_t lo = sm_dataEveryThreadStartIdx;
		size_t hi = lo + num - 1;

		quickSort(lo, hi, sm_dataBegin, sm_stack, sm_posBegin);
	}

	__syncthreads();


	// 将排序好的数据从share memory中拷贝到global memory
	for (int i = 0; i < readTime; i++)
	{
	    addrIdx = i * blockDim.x + tid;
		if (addrIdx < blockHandleNum)
		{
			segData[everyBlockStartDataIdx + addrIdx] =  sm_dataBegin[addrIdx];
		}
	}

	__syncthreads();

	// 将排序号的位置从share memory中拷贝到global memory
	for (int i = 0; i < readTime; i++)
	{
	    addrIdx = i * blockDim.x + tid;
		if (addrIdx < blockHandleNum)
		{
			segPos[everyBlockStartDataIdx + addrIdx] 
				=  sm_posBegin[addrIdx] + segPos[everyBlockStartDataIdx + addrIdx];
		}
	}

	__syncthreads();

	//}

    
}


#endif