#include "kdgpu_data.h"


// ****************³ÉÔ±º¯Êý******************



/**
 * @brife init prims from orgin format to formal
 * 
 * @param[in] primsNum number of prims
 * @param[in] prims in orgin state
**/
void Prims::initPrims(SceneInfoArr &scene, int keynum)
{
    for (int i = 0; i < keynum; i++)
	{
		//// face
		vertexNum_[i] = scene.vertexNum[i];
		faceNum_[i] = scene.faceNum[i];
		h_vertex_[i] = scene.h_vertex[i];
		h_face_[i] = scene.h_face[i];
		
		// alloc
		cudaMalloc((void **)&d_vertex_[i], sizeof(float4) * vertexNum_[i]);
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_face_[i], sizeof(FaceType) * faceNum_[i]));

		// cpy
		CUDA_SAFE_CALL(cudaMemcpy(d_vertex_[i], scene.h_vertex[i], sizeof(VertexType) * vertexNum_[i], cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_face_[i], scene.h_face[i], sizeof(FaceType) * faceNum_[i], cudaMemcpyHostToDevice));

		free(scene.h_vertex[i]);
		free(scene.h_face[i]);

#ifdef NEED_TEXTURE
		//// texture
		textureNum_[i] = scene.textureNum[i];
		faceTextureNum_[i] = scene.faceTextureNum[i];
		h_texture_[i] = scene.h_texture[i];
		h_facetexture_[i] = scene.h_facetexture[i];

		CUDA_SAFE_CALL(cudaMalloc((void **)&d_texture_[i], sizeof(TextureType) * textureNum_[i]));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_facetexture_[i], sizeof(FaceVTType) * faceTextureNum_[i]));
		CUDA_SAFE_CALL(cudaMemcpy(d_texture_[i], h_texture_[i], sizeof(TextureType) * textureNum_[i], cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_facetexture_[i], h_facetexture_[i], sizeof(FaceVTType) * faceTextureNum_[i], cudaMemcpyHostToDevice));

		free(scene.h_texture[i]);
		free(scene.h_facetexture[i]);
#endif

#ifdef NEED_NORMAL
		//// normal
		normalNum_[i] = scene.normalNum[i];
		faceNormalNum_[i] = scene.faceNormalNum[i];
		h_normal_[i] = scene.h_normal[i];
		h_facenormal_[i] = scene.h_facenormal[i];

		CUDA_SAFE_CALL(cudaMalloc((void **)&d_normal_[i], sizeof(NormalType) * normalNum_[i]));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_facenormal_[i], sizeof(FaceVNType) * faceNormalNum_[i]));
		CUDA_SAFE_CALL(cudaMemcpy(d_normal_[i], h_normal_[i], sizeof(NormalType) * normalNum_[i], cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_facenormal_[i], h_facenormal_[i], sizeof(FaceVNType) * faceNormalNum_[i], cudaMemcpyHostToDevice));

		free(scene.h_normal[i]);
		free(scene.h_facenormal[i]);

#endif 
	}
	

}


void AllBoundingBoxList::initPrimsBoundingBox(size_t nPrims)
{
	size_ = nPrims;
    CUDA_SAFE_CALL(cudaMalloc((void**)&primsBoundingBox_[0], sizeof(float4) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&primsBoundingBox_[1], sizeof(float4) * size_));
}

void AllBoundingBoxList::realloc(size_t referSize, bool needCpy)
{
	// needn't realloc mem
	if (referSize <= size_) return;

	// save old size and mem
	float4 *oldMem[2];
	oldMem[0] = primsBoundingBox_[0];
	oldMem[1] = primsBoundingBox_[1];

	size_t oldSize = size_;

	//  cal new size
    while(size_ < referSize)
	{
	    size_ <<= 1;
	}
	
	// alloc new mem
	CUDA_SAFE_CALL(cudaMalloc((void**)&primsBoundingBox_[0], sizeof(float4) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&primsBoundingBox_[1], sizeof(float4) * size_));

	// memcpy
	if (needCpy)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(primsBoundingBox_[0], oldMem[0], sizeof(float4) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(primsBoundingBox_[1], oldMem[1], sizeof(float4) * oldSize, cudaMemcpyDeviceToDevice));
	}

	// release old mem
	CUDA_SAFE_CALL(cudaFree(oldMem[0]));
	CUDA_SAFE_CALL(cudaFree(oldMem[1]));


}

void TriNodePrimsBBList::initTriNodePrimsBBList(size_t capacity)
{
	/*capacity_ = capacity;
	size_ = 0;*/
	size_ = capacity;
    
	// alloc
	/*CUDA_SAFE_CALL(cudaMalloc((void**)&d_capacity_, sizeof(size_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_size_, sizeof(size_t)));*/
    CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[0], sizeof(float) * capacity));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[1], sizeof(float) * capacity));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[2], sizeof(float) * capacity));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[3], sizeof(float) * capacity));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[4], sizeof(float) * capacity));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[5], sizeof(float) * capacity));

	// trans
	/*CUDA_SAFE_CALL(cudaMemcpy(d_capacity_, &capacity_, sizeof(size_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_size_, &size_, sizeof(size_t), cudaMemcpyHostToDevice));*/
}

/*size_t getSize()
{
    CUDA_SAFE_CALL(cudaMemcpy(&size_, d_size_, sizeof(size_t), cudaMemcpyDeviceToHost));
	return size_;
}*/

void TriNodePrimsBBList::realloc(size_t referSize, bool needCpy)
{
    // needn't realloc mem
	if (referSize <= size_) return;

	// save old size and mem
	float *oldMem[6];
	oldMem[0] = triNodePrimsBBList_[0];
	oldMem[1] = triNodePrimsBBList_[1];
	oldMem[2] = triNodePrimsBBList_[2];
	oldMem[3] = triNodePrimsBBList_[3];
	oldMem[4] = triNodePrimsBBList_[4];
	oldMem[5] = triNodePrimsBBList_[5];

	size_t oldSize = size_;

	//  cal new size
    while(size_ < referSize)
	{
	    size_ <<= 1;
	}
	
	// alloc new mem
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[0], sizeof(float) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[1], sizeof(float) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[2], sizeof(float) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[3], sizeof(float) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[4], sizeof(float) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&triNodePrimsBBList_[5], sizeof(float) * size_));

	// memcpy
	if (needCpy)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[0], oldMem[0], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[1], oldMem[1], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[2], oldMem[2], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[3], oldMem[3], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[4], oldMem[4], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(triNodePrimsBBList_[5], oldMem[5], sizeof(float) * oldSize, cudaMemcpyDeviceToDevice));

	}

	// release old mem
	CUDA_SAFE_CALL(cudaFree(oldMem[0]));
	CUDA_SAFE_CALL(cudaFree(oldMem[1]));
	CUDA_SAFE_CALL(cudaFree(oldMem[2]));
	CUDA_SAFE_CALL(cudaFree(oldMem[3]));
	CUDA_SAFE_CALL(cudaFree(oldMem[4]));
	CUDA_SAFE_CALL(cudaFree(oldMem[5]));

}






void TriNodeList::initTriNodeList(size_t num)
{
    size_ = num;
	CUDA_SAFE_CALL(cudaMalloc((void**)&triIdx_, sizeof(size_t) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&segFlags_, sizeof(size_t) * size_));
}

void TriNodeList::realloc(size_t referSize, bool needCpy)
{
	// needn't realloc mem
	if (referSize <= size_) return;

	// save old size and mem
	size_t *oldMem[2];
	oldMem[0] = triIdx_;
	oldMem[1] = segFlags_;

	size_t oldSize = size_;

	//  cal new size
    while(size_ < referSize)
	{
	    size_ <<= 1;
	}
	
	// alloc new mem
	CUDA_SAFE_CALL(cudaMalloc((void**)&triIdx_, sizeof(size_t) * size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&segFlags_, sizeof(size_t) * size_));

	// memcpy
	if (needCpy)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(triIdx_, oldMem[0], sizeof(size_t) * oldSize, cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(segFlags_, oldMem[1], sizeof(size_t) * oldSize, cudaMemcpyDeviceToDevice));
	}

	// release old mem
	CUDA_SAFE_CALL(cudaFree(oldMem[0]));
	CUDA_SAFE_CALL(cudaFree(oldMem[1]));


}