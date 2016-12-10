

//#include "kdgpu_kernal.h"
//#ifndef _KDGPU_APP_CU_
//#define _KDGPU_APP_CU_
#include "kdgpu_app.h"
#include "base.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudpp.h"
#include "device_launch_parameters.h"

#include "png_image.h"

#include <stack>
#include <time.h>
#include <vector>


#include "kdgpu_data.h"
#include "cutimer.h"

#include "kdgpu_kernal.cu"
#include "raytracing_kernal.cu"
#include "seg_radix_sort_kernal.cu"


using namespace std;
/************************************************************************************************/
/*                                       HOST                                                   */
/************************************************************************************************/
void GPU_BuildKdTree::globalInit(SceneInfoArr &scene, int keynum, 
									 uchar4* h_gtex, uint4* h_texPos)
{
    prims_.initPrims(scene, keynum);

	kdNodeBase_.list_[0] = NULL;
	kdNodeExtra_.list_[0] = NULL;
	kdNodeBB_.list_[0] = NULL;
	kdNodeBase_.size_ = 0;


	iCost_ = 80.0f;
	tCost_ = 1.0f;

#ifdef NEED_TEXTURE
	// bind texture
	bindTexture2D(h_gtex, h_texPos);
#endif

	//// trans mask info to device
	// h_rootMaskHi: 根据small node中tris的数目来确定原始small node的位掩码的高位字节，比如说1对应着0x80000000
	size_t h_rootMaskHi[64] = { 0x80000000, 0xc0000000, 0xe0000000, 0xf0000000, 0xf8000000, 0xfc000000, 0xfe000000, 0xff000000, 
		0xff800000, 0xffc00000, 0xffe00000, 0xfff00000, 0xfff80000, 0xfffc0000, 0xfffe0000, 0xffff0000, 
		0xffff8000, 0xffffc000, 0xffffe000, 0xfffff000, 0xfffff800, 0xfffffc00, 0xfffffe00, 0xffffff00, 
		0xffffff80, 0xffffffc0, 0xffffffe0, 0xfffffff0, 0xfffffff8, 0xfffffffc, 0xfffffffe, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
	
	// 原始small node位掩码的地位字节
	size_t h_rootMaskLo[64] = { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x80000000, 0xc0000000, 0xe0000000, 0xf0000000, 0xf8000000, 0xfc000000, 0xfe000000, 0xff000000, 
		0xff800000, 0xffc00000, 0xffe00000, 0xfff00000, 0xfff80000, 0xfffc0000, 0xfffe0000, 0xffff0000, 
		0xffff8000, 0xffffc000, 0xffffe000, 0xfffff000, 0xfffff800, 0xfffffc00, 0xfffffe00, 0xffffff00, 
		0xffffff80, 0xffffffc0, 0xffffffe0, 0xfffffff0, 0xfffffff8, 0xfffffffc, 0xfffffffe, 0xffffffff};
	
	// 数组中的第i个元素(从1开始)表示的是：第i个物体包围盒的左边界如果作为split plane时，该处的位掩码的高位
	// 表示tri：right------>left
	size_t h_rootStartMaskHi[64] = {	0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000, 
		0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000, 
		0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100, 
		0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
	
	// 位掩码的地位
	size_t h_rootStartMaskLo[64]  =  {	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
		0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000, 
		0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000, 
		0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100, 
		0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001};

	// 数组中的第i个(从1开始)表示的是：第i个物体包围盒的右边界做为split plane时，该处的位掩码的高位
	// 表示tri: left----->right
	size_t h_rootEndMaskHi[64] = {	0x7fffffff, 0xbfffffff, 0xdfffffff, 0xefffffff, 0xf7ffffff, 0xfbffffff, 0xfdffffff, 0xfeffffff, 
		0xff7fffff, 0xffbfffff, 0xffdfffff, 0xffefffff, 0xfff7ffff, 0xfffbffff, 0xfffdffff, 0xfffeffff, 
		0xffff7fff, 0xffffbfff, 0xffffdfff, 0xffffefff, 0xfffff7ff, 0xfffffbff, 0xfffffdff, 0xfffffeff, 
		0xffffff7f, 0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7, 0xfffffffb, 0xfffffffd, 0xfffffffe, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
	size_t h_rootEndMaskLo[64] = {	0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 
		0x7fffffff, 0xbfffffff, 0xdfffffff, 0xefffffff, 0xf7ffffff, 0xfbffffff, 0xfdffffff, 0xfeffffff, 
		0xff7fffff, 0xffbfffff, 0xffdfffff, 0xffefffff, 0xfff7ffff, 0xfffbffff, 0xfffdffff, 0xfffeffff, 
		0xffff7fff, 0xffffbfff, 0xffffdfff, 0xffffefff, 0xfffff7ff, 0xfffffbff, 0xfffffdff, 0xfffffeff, 
		0xffffff7f, 0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7, 0xfffffffb, 0xfffffffd, 0xfffffffe};

	// 0~255之间数的二进制表示1的个数, 用于bit counting
	char h_bits_in_char[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
		4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

	// 0~255之间的数的二进制表示1所在的位置(256 * 8 = 2048), 每个数目有8个位置
	// 从左至右，e.g.5的二进制是00000101, 1的数目有2个，分别在5,7位置上，因此对应5 7 0 0 0 0 0 0 
	char h_one_loc[2048] = {0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0, 
		4, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 0, 4, 6, 7, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 4, 5, 7, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 
		3, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 3, 6, 7, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 0, 0, 0, 3, 5, 6, 0, 0, 0, 0, 0, 3, 5, 6, 7, 0, 0, 0, 0, 
		3, 4, 0, 0, 0, 0, 0, 0, 3, 4, 7, 0, 0, 0, 0, 0, 3, 4, 6, 0, 0, 0, 0, 0, 3, 4, 6, 7, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 3, 4, 5, 7, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 3, 4, 5, 6, 7, 0, 0, 0, 
		2, 0, 0, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 2, 6, 7, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 2, 5, 7, 0, 0, 0, 0, 0, 2, 5, 6, 0, 0, 0, 0, 0, 2, 5, 6, 7, 0, 0, 0, 0, 
		2, 4, 0, 0, 0, 0, 0, 0, 2, 4, 7, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0, 2, 4, 6, 7, 0, 0, 0, 0, 2, 4, 5, 0, 0, 0, 0, 0, 2, 4, 5, 7, 0, 0, 0, 0, 2, 4, 5, 6, 0, 0, 0, 0, 2, 4, 5, 6, 7, 0, 0, 0, 
		2, 3, 0, 0, 0, 0, 0, 0, 2, 3, 7, 0, 0, 0, 0, 0, 2, 3, 6, 0, 0, 0, 0, 0, 2, 3, 6, 7, 0, 0, 0, 0, 2, 3, 5, 0, 0, 0, 0, 0, 2, 3, 5, 7, 0, 0, 0, 0, 2, 3, 5, 6, 0, 0, 0, 0, 2, 3, 5, 6, 7, 0, 0, 0, 
		2, 3, 4, 0, 0, 0, 0, 0, 2, 3, 4, 7, 0, 0, 0, 0, 2, 3, 4, 6, 0, 0, 0, 0, 2, 3, 4, 6, 7, 0, 0, 0, 2, 3, 4, 5, 0, 0, 0, 0, 2, 3, 4, 5, 7, 0, 0, 0, 2, 3, 4, 5, 6, 0, 0, 0, 2, 3, 4, 5, 6, 7, 0, 0, 
		1, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 1, 6, 7, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 1, 5, 7, 0, 0, 0, 0, 0, 1, 5, 6, 0, 0, 0, 0, 0, 1, 5, 6, 7, 0, 0, 0, 0, 
		1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 7, 0, 0, 0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 0, 1, 4, 6, 7, 0, 0, 0, 0, 1, 4, 5, 0, 0, 0, 0, 0, 1, 4, 5, 7, 0, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 0, 1, 4, 5, 6, 7, 0, 0, 0, 
		1, 3, 0, 0, 0, 0, 0, 0, 1, 3, 7, 0, 0, 0, 0, 0, 1, 3, 6, 0, 0, 0, 0, 0, 1, 3, 6, 7, 0, 0, 0, 0, 1, 3, 5, 0, 0, 0, 0, 0, 1, 3, 5, 7, 0, 0, 0, 0, 1, 3, 5, 6, 0, 0, 0, 0, 1, 3, 5, 6, 7, 0, 0, 0, 
		1, 3, 4, 0, 0, 0, 0, 0, 1, 3, 4, 7, 0, 0, 0, 0, 1, 3, 4, 6, 0, 0, 0, 0, 1, 3, 4, 6, 7, 0, 0, 0, 1, 3, 4, 5, 0, 0, 0, 0, 1, 3, 4, 5, 7, 0, 0, 0, 1, 3, 4, 5, 6, 0, 0, 0, 1, 3, 4, 5, 6, 7, 0, 0, 
		1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 7, 0, 0, 0, 0, 0, 1, 2, 6, 0, 0, 0, 0, 0, 1, 2, 6, 7, 0, 0, 0, 0, 1, 2, 5, 0, 0, 0, 0, 0, 1, 2, 5, 7, 0, 0, 0, 0, 1, 2, 5, 6, 0, 0, 0, 0, 1, 2, 5, 6, 7, 0, 0, 0, 
		1, 2, 4, 0, 0, 0, 0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 1, 2, 4, 6, 0, 0, 0, 0, 1, 2, 4, 6, 7, 0, 0, 0, 1, 2, 4, 5, 0, 0, 0, 0, 1, 2, 4, 5, 7, 0, 0, 0, 1, 2, 4, 5, 6, 0, 0, 0, 1, 2, 4, 5, 6, 7, 0, 0, 
		1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 7, 0, 0, 0, 0, 1, 2, 3, 6, 0, 0, 0, 0, 1, 2, 3, 6, 7, 0, 0, 0, 1, 2, 3, 5, 0, 0, 0, 0, 1, 2, 3, 5, 7, 0, 0, 0, 1, 2, 3, 5, 6, 0, 0, 0, 1, 2, 3, 5, 6, 7, 0, 0, 
		1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4, 7, 0, 0, 0, 1, 2, 3, 4, 6, 0, 0, 0, 1, 2, 3, 4, 6, 7, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 7, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 
		0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 0, 4, 6, 7, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 4, 5, 7, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 
		0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 3, 6, 7, 0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 0, 0, 3, 5, 7, 0, 0, 0, 0, 0, 3, 5, 6, 0, 0, 0, 0, 0, 3, 5, 6, 7, 0, 0, 0, 
		0, 3, 4, 0, 0, 0, 0, 0, 0, 3, 4, 7, 0, 0, 0, 0, 0, 3, 4, 6, 0, 0, 0, 0, 0, 3, 4, 6, 7, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 3, 4, 5, 7, 0, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 3, 4, 5, 6, 7, 0, 0, 
		0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 2, 6, 7, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 2, 5, 7, 0, 0, 0, 0, 0, 2, 5, 6, 0, 0, 0, 0, 0, 2, 5, 6, 7, 0, 0, 0, 
		0, 2, 4, 0, 0, 0, 0, 0, 0, 2, 4, 7, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0, 2, 4, 6, 7, 0, 0, 0, 0, 2, 4, 5, 0, 0, 0, 0, 0, 2, 4, 5, 7, 0, 0, 0, 0, 2, 4, 5, 6, 0, 0, 0, 0, 2, 4, 5, 6, 7, 0, 0, 
		0, 2, 3, 0, 0, 0, 0, 0, 0, 2, 3, 7, 0, 0, 0, 0, 0, 2, 3, 6, 0, 0, 0, 0, 0, 2, 3, 6, 7, 0, 0, 0, 0, 2, 3, 5, 0, 0, 0, 0, 0, 2, 3, 5, 7, 0, 0, 0, 0, 2, 3, 5, 6, 0, 0, 0, 0, 2, 3, 5, 6, 7, 0, 0, 
		0, 2, 3, 4, 0, 0, 0, 0, 0, 2, 3, 4, 7, 0, 0, 0, 0, 2, 3, 4, 6, 0, 0, 0, 0, 2, 3, 4, 6, 7, 0, 0, 0, 2, 3, 4, 5, 0, 0, 0, 0, 2, 3, 4, 5, 7, 0, 0, 0, 2, 3, 4, 5, 6, 0, 0, 0, 2, 3, 4, 5, 6, 7, 0, 
		0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 1, 6, 7, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 1, 5, 7, 0, 0, 0, 0, 0, 1, 5, 6, 0, 0, 0, 0, 0, 1, 5, 6, 7, 0, 0, 0, 
		0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 7, 0, 0, 0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 0, 1, 4, 6, 7, 0, 0, 0, 0, 1, 4, 5, 0, 0, 0, 0, 0, 1, 4, 5, 7, 0, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 0, 1, 4, 5, 6, 7, 0, 0, 
		0, 1, 3, 0, 0, 0, 0, 0, 0, 1, 3, 7, 0, 0, 0, 0, 0, 1, 3, 6, 0, 0, 0, 0, 0, 1, 3, 6, 7, 0, 0, 0, 0, 1, 3, 5, 0, 0, 0, 0, 0, 1, 3, 5, 7, 0, 0, 0, 0, 1, 3, 5, 6, 0, 0, 0, 0, 1, 3, 5, 6, 7, 0, 0, 
		0, 1, 3, 4, 0, 0, 0, 0, 0, 1, 3, 4, 7, 0, 0, 0, 0, 1, 3, 4, 6, 0, 0, 0, 0, 1, 3, 4, 6, 7, 0, 0, 0, 1, 3, 4, 5, 0, 0, 0, 0, 1, 3, 4, 5, 7, 0, 0, 0, 1, 3, 4, 5, 6, 0, 0, 0, 1, 3, 4, 5, 6, 7, 0, 
		0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 7, 0, 0, 0, 0, 0, 1, 2, 6, 0, 0, 0, 0, 0, 1, 2, 6, 7, 0, 0, 0, 0, 1, 2, 5, 0, 0, 0, 0, 0, 1, 2, 5, 7, 0, 0, 0, 0, 1, 2, 5, 6, 0, 0, 0, 0, 1, 2, 5, 6, 7, 0, 0, 
		0, 1, 2, 4, 0, 0, 0, 0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 1, 2, 4, 6, 0, 0, 0, 0, 1, 2, 4, 6, 7, 0, 0, 0, 1, 2, 4, 5, 0, 0, 0, 0, 1, 2, 4, 5, 7, 0, 0, 0, 1, 2, 4, 5, 6, 0, 0, 0, 1, 2, 4, 5, 6, 7, 0, 
		0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 7, 0, 0, 0, 0, 1, 2, 3, 6, 0, 0, 0, 0, 1, 2, 3, 6, 7, 0, 0, 0, 1, 2, 3, 5, 0, 0, 0, 0, 1, 2, 3, 5, 7, 0, 0, 0, 1, 2, 3, 5, 6, 0, 0, 0, 1, 2, 3, 5, 6, 7, 0, 
		0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4, 7, 0, 0, 0, 1, 2, 3, 4, 6, 0, 0, 0, 1, 2, 3, 4, 6, 7, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 7, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3, 4, 5, 6, 7, 
	};


	// 把掩码拷贝到显存的constant memory中
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootMaskHi_, h_rootMaskHi, sizeof(size_t) * 64));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootMaskLo_, h_rootMaskLo, sizeof(size_t) * 64));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootStartMaskHi_, h_rootStartMaskHi, sizeof(size_t) * 64));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootStartMaskLo_, h_rootStartMaskLo, sizeof(size_t) * 64));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootEndMaskHi_, h_rootEndMaskHi, sizeof(size_t) * 64));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)d_rootEndMaskLo_, h_rootEndMaskLo, sizeof(size_t) * 64));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_bits_in_char, h_bits_in_char, sizeof(char) * 256));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_one_loc, h_one_loc, sizeof(char) * 2048));

    kdScan(	CUDPP_ADD, 
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
			NULL,
			NULL,
			0);

	keyframe_ = 0;
}

void GPU_BuildKdTree::initialization()
{
	allocVideoMem_ = 0;

	nPrims_ = prims_.faceNum_[keyframe_];
	nVertex_ = prims_.vertexNum_[keyframe_];

	// estimate
	size_t kdNodeCapacity				= 2 * nPrims_; // 2 * nPrims_
	size_t nextListCapacity				= nPrims_;
	size_t triNodeListCapacity			= 2 * nPrims_; 
	size_t allBoundingBoxCapacity		= nPrims_;
	size_t triNodePrimsBBListCapacity	= triNodeListCapacity;
	size_t nextNodeNumListCapacity		= nextListCapacity;
	size_t childListFlagsCapacity		= triNodeListCapacity;

	size_t largeNodeNextListCapacity	= nextListCapacity;
	size_t largeNodeTriNodeListCapacity	= triNodeListCapacity;
	size_t largeNodeFlagsListCapacity	= nextListCapacity;
	size_t largeNodeNumListCapacity		= largeNodeFlagsListCapacity;
	size_t largeNodeTriNodeFlagsListCapacity = triNodeListCapacity;

	size_t smallNodeNextListCapacity	= nPrims_;
	size_t smallNodeTriNodeListCapacity = nPrims_;
	size_t smallNodeFlagsListCapacity   = nextListCapacity;
	size_t smallNodeNumListCapacity		= smallNodeFlagsListCapacity;
	size_t smallNodeTriNodeFlagsListCapacity = triNodeListCapacity;


	// kdNode_ 
	//~kdNode_.initList(kdNodeCapacity);
	//~allocVideoMem_ += sizeof(KdNodeOrigin) * kdNodeCapacity;
	kdNodeBase_.initList(kdNodeCapacity);
	kdNodeBB_.initList(kdNodeCapacity);
	kdNodeExtra_.initList(kdNodeCapacity);
	allocVideoMem_ += (sizeof(KdNode_base) + sizeof(KdNode_bb) + sizeof(KdNode_extra)) * kdNodeCapacity;

	// activeList_

	// nextList_
	nextList_.initList(nextListCapacity);
	allocVideoMem_ += sizeof(size_t) * nextListCapacity;

	// triNodeList_
	triNodeList_.initList(triNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * triNodeListCapacity;

	// allBoundingBoxList_
	allBoundingBoxList_.initList(allBoundingBoxCapacity);
	allocVideoMem_ += 2 * sizeof(float4) * allBoundingBoxCapacity;

	// newTriNodeList_
	newTriNodeList_.initList(triNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * triNodeListCapacity;

	// largeNodeNextList_
	largeNodeNextList_.initList(largeNodeNextListCapacity);
	allocVideoMem_ += sizeof(int *) * largeNodeNextListCapacity;

	// largeNodeTriNodeList_
	largeNodeTriNodeList_.initList(largeNodeTriNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * largeNodeTriNodeListCapacity;

	// smallNodeNextList_
	smallNodeNextList_.initList(smallNodeNextListCapacity);
	allocVideoMem_ += sizeof(int *) * smallNodeNextListCapacity;

	// smallNodeTriNodeList_
	smallNodeTriNodeList_.initList(smallNodeTriNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * smallNodeTriNodeListCapacity;

	d_itriNodePrimsBBListAssist_.initList(triNodePrimsBBListCapacity);
	d_otriNodePrimsBBListAssist_.initList(triNodePrimsBBListCapacity);

	allocVideoMem_ += 12 * sizeof(float) * triNodePrimsBBListCapacity;
	d_isplitListAssist_.initList(triNodeListCapacity);
	d_osplitListAssist_.initList(triNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(float) * triNodeListCapacity;
	
	// splitTypeListAssist_
	d_isplitTypeListAssist_.initList(triNodeListCapacity);
	d_osplitTypeListAssist_.initList(triNodeListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * triNodeListCapacity;

    // leftChildListFlagsAssist_
	d_ileftChildListFlagsAssist_.initList(childListFlagsCapacity);
	d_oleftChildListFlagsAssist_.initList(childListFlagsCapacity);

	// rightChildListFlagsAssist_
	d_irightChildListFlagsAssist_.initList(childListFlagsCapacity);
	d_orightChildListFlagsAssist_.initList(childListFlagsCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * childListFlagsCapacity;

    // nextNodeNumListAssist_
	d_inextNodeNumListAssist_.initList(nextNodeNumListCapacity);
	d_onextNodeNumListAssist_.initList(nextNodeNumListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * nextNodeNumListCapacity;

	// largeNodeFlagsListAssist_
	d_ilargeNodeFlagsListAssist_.initList(largeNodeFlagsListCapacity);
	d_olargeNodeFlagsListAssist_.initList(largeNodeFlagsListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * largeNodeFlagsListCapacity;
	
	// largeNodeNumListAssist_
	d_ilargeNodeNumListAssist_.initList(largeNodeNumListCapacity);
	d_olargeNodeNumListAssist_.initList(largeNodeNumListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * largeNodeFlagsListCapacity;
	
	// largeNodeTriNodeFlagsListAssist_
	d_ilargeNodeTriNodeFlagsListAssist_.initList(largeNodeTriNodeFlagsListCapacity);
	d_olargeNodeTriNodeFlagsListAssist_.initList(largeNodeTriNodeFlagsListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * largeNodeTriNodeFlagsListCapacity;
 
    // smallNodeFlagsListAssist_
	d_ismallNodeFlagsListAssist_.initList(smallNodeFlagsListCapacity);
	d_osmallNodeFlagsListAssist_.initList(smallNodeFlagsListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * smallNodeFlagsListCapacity;
	
	// smallNodeNumListAssist_
	d_ismallNodeNumListAssist_.initList(smallNodeNumListCapacity);
    d_osmallNodeNumListAssist_.initList(smallNodeNumListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * smallNodeNumListCapacity; 
	
	// smallNodeTriNodeFlagsListAssist_
	d_ismallNodeTriNodeFlagsListAssist_.initList(smallNodeTriNodeFlagsListCapacity);
	d_osmallNodeTriNodeFlagsListAssist_.initList(smallNodeTriNodeFlagsListCapacity);
	allocVideoMem_ += 2 * sizeof(size_t) * smallNodeTriNodeFlagsListCapacity;

	// numValid
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_numValid_, sizeof(size_t)));
	allocVideoMem_ += sizeof(size_t);
	h_numValid_ = (size_t *)malloc(sizeof(size_t));


	smallNodeNextListSizeGlobal_= 0;
	smallNodeTriNodeListSizeGlobal_ = 0;

	leafNum_ = 0;

	iCost_ = 80.0f;
	tCost_ = 1.0f;

	
	h_pixelBuf_ = NULL;
	d_pixelBuf_ = NULL;

}

#ifdef NEED_TEXTURE
void GPU_BuildKdTree::bindTexture2D(uchar4* h_gtex, uint4* h_texPos)
{
    // bind h_tex
	int s = TEX_G_WID * TEX_G_HEI * sizeof(uchar4);
	
	// channel desc
	cudaChannelFormatDesc channelUchar4
		= cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned ); 
	

	// cuda array
	cudaArray* cuArr_tex;
	cudaMallocArray( &cuArr_tex, &channelUchar4, TEX_G_WID, TEX_G_HEI );

	/*CuTimer texTimer;
	texTimer.startTimer();*/
	cudaMemcpyToArray( cuArr_tex, 0, 0, h_gtex, s, cudaMemcpyHostToDevice );
	//texTimer.finishTimer("cudaMemcpyToArray ");
	
	// tex init
	texKa.normalized	= 0;
	texKa.filterMode	= cudaFilterModePoint;
	texKa.addressMode[0] = cudaAddressModeClamp;
	texKa.channelDesc = channelUchar4;

	// bind
	cudaBindTextureToArray( texKa, cuArr_tex, channelUchar4 );

	// release
	delete [] h_gtex;
	delete [] h_texPos;

	// 
	cudaMemcpyToSymbol((char*)d_texPos, h_texPos, sizeof(uint4) * TEXMAX);
	

}

void GPU_BuildKdTree::testTexture(Tex* tex, int ppmNum)
{
	// 纹理测试
	//for (int i = 0; i < ppmNum; i++)
	//{
	//    
	//	size_t width = tex[i].wid;
	//	size_t height = tex[i].hei;

	//	//*/
	//	size_t blockSize = 64;
	//	dim3 g(width * height / blockSize + 1, 1, 1);
	//	dim3 b(blockSize, 1, 1);
	//	printf("gridDim: (%d, %d, %d), blockDim: (%d, %d, %d)\n", g.x, g.y, g.z, b.x, b.y, b.z);
	//	uchar4* d_tex;
	//	cudaMalloc((void **)&d_tex, sizeof(uchar4) * width * height);
	//	testTextureBind<<<g, b, 0>>>(width, height, d_tex, i);
	//	uchar4* h_tex = (uchar4 *)malloc(sizeof(uchar4) * width * height);
	//	cudaMemcpy( h_tex, d_tex, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost );

	//	//writepng("E:\\textestresult\\ga_origin.png", width, height, (unsigned char*)tex[0].texPtr);
	//	char mainpath[50] = "E:\\xialong\\dynamic\\testtexture\\";
	//	char path[100];
	//	sprintf(path, "%s%d.png", mainpath, i);
	//	writepng(path, width, height, (unsigned char *)h_tex);
	//}
}

#endif

#ifdef NEED_TEXTURE

#endif

// 释放处理large node时开辟的空间
void GPU_BuildKdTree::largeNodeRelease()
{
	activeList_.release();

	nextList_.release();

	// triNodeList_
	triNodeList_.release();

	// allBoundingBoxList_

	// newTriNodeList_
	newTriNodeList_.release();

	// largeNodeNextList_
	largeNodeNextList_.release();

	// largeNodeTriNodeList_
	largeNodeTriNodeList_.release();

	// smallNodeNextList_

	// smallNodeTriNodeList_

	d_itriNodePrimsBBListAssist_.release();
    d_otriNodePrimsBBListAssist_.release();

	// splitListAssist_
    d_isplitListAssist_.release();
    d_osplitListAssist_.release();
	
	// splitTypeListAssist_
    d_isplitTypeListAssist_.release();
    d_osplitTypeListAssist_.release();

    // leftChildListFlagsAssist_
    d_ileftChildListFlagsAssist_.release();
    d_oleftChildListFlagsAssist_.release();

	// rightChildListFlagsAssist_
    d_irightChildListFlagsAssist_.release();
    d_orightChildListFlagsAssist_.release();

    // nextNodeNumListAssist_
	d_inextNodeNumListAssist_.release();
    d_onextNodeNumListAssist_.release();

	// largeNodeFlagsListAssist_
    d_ilargeNodeFlagsListAssist_.release();
    d_olargeNodeFlagsListAssist_.release();
	
	// largeNodeNumListAssist_
    d_ilargeNodeNumListAssist_.release();
	d_olargeNodeNumListAssist_.release();
	
	// largeNodeTriNodeFlagsListAssist_
    d_ilargeNodeTriNodeFlagsListAssist_.release();
    d_olargeNodeTriNodeFlagsListAssist_.release();
 
    // smallNodeFlagsListAssist_
    d_ismallNodeFlagsListAssist_.release();
    d_osmallNodeFlagsListAssist_.release();
	
	// smallNodeNumListAssist_
    d_ismallNodeNumListAssist_.release();
    d_osmallNodeNumListAssist_.release();
	
	// smallNodeTriNodeFlagsListAssist_
    d_ismallNodeTriNodeFlagsListAssist_.release();
    d_osmallNodeTriNodeFlagsListAssist_.release();

	// numValid
	
}


// 释放处理small node的显存资源
void GPU_BuildKdTree::smallNodeRelease()
{
	//cudaError_t e;

	smallNodeList_.release();
// printf("kaka0\n");
	smallNodeNextList_.release();
	
	smallNodeTriNodeList_.release();
// printf("kaka1\n");
	allBoundingBoxList_.release();

    d_ismallNodeRootList_.release();
	d_osmallNodeRootList_.release();

	smallNodeRootMaskHi_.release();
    smallNodeRootMaskLo_.release();

    smallNodeBoundryFlags_.release();

    smallNodeBoundryValue_.release();
    smallNodeBoundryRPos_.release();
    smallNodeBoundryType_.release();
    smallNodeBoundryTriIdx_.release();
    smallNodeBoundryAPos_.release();
// printf("kaka2\n");
    d_ismallNodeEveryLeafSize_.release();
	d_osmallNodeEveryLeafSize_.release();
	d_ismallNodeSegStartAddr_.release();
	d_osmallNodeSegStartAddr_.release();

    d_ismallNodeMaskLeftHi_.release();
    d_osmallNodeMaskLeftHi_.release();
    d_ismallNodeMaskLeftLo_.release();
    d_osmallNodeMaskLeftLo_.release();
	d_ismallNodeMaskRightHi_.release();
	d_osmallNodeMaskRightHi_.release();
    d_ismallNodeMaskRightLo_.release();
    d_osmallNodeMaskRightLo_.release();
 
// printf("kaka3\n");
	
	smallNodeMaskHi_.release();
    smallNodeMaskLo_.release();

	smallNodeNextListMaskHi_.release();
    smallNodeNextListMaskLo_.release();

	smallNodeRList_.release();

    d_ismallNodeLeafFlags_.release();
    d_osmallNodeLeafFlags_.release();

    d_ismallNodeNoLeafFlags_.release();
    d_osmallNodeNoLeafFlags_.release();

    d_ismallNodeLeafSize_.release();
    d_osmallNodeLeafSize_.release();

	cudaFree(d_numValid_);
	free(h_numValid_);

	// forget release
	smallNodeNextListRList_.release();
	
// printf("kaka4\n");
}

//void GPU_BuildKdTree::saveKdTree()
//{
//    FILE* fp;
//	fp = fopen("C:\\kdtree.tree", "wb+");
//	KdNodeOrigin *h_kdNode = (KdNodeOrigin *)malloc(sizeof(KdNodeOrigin) * kdNode_.size_);
//	size_t *leafTriIndex = (size_t *)malloc(sizeof(size_t) * leafPrims_.size_);
//	CUDA_SAFE_CALL(cudaMemcpy(leafTriIndex, leafPrims_.list_[0], sizeof(size_t) * leafPrims_.size_, cudaMemcpyDeviceToHost));
//	CUDA_SAFE_CALL(cudaMemcpy(h_kdNode, kdNode_.list_[0], sizeof(KdNodeOrigin) * kdNode_.size_, cudaMemcpyDeviceToHost));
//
//	// save tree node
//	fwrite(&kdNode_.size_, sizeof(size_t), 1, fp);
//	fwrite(h_kdNode, sizeof(KdNodeOrigin), kdNode_.size_, fp);
//
//	// save leaf tris index
//	fwrite(&leafPrims_.size_, sizeof(size_t), 1, fp);
//	fwrite(leafTriIndex, sizeof(size_t), leafPrims_.size_, fp);
//
//	free(h_kdNode);
//	free(leafTriIndex);
//
//	fclose(fp);
//	
//}

//void GPU_BuildKdTree::readKdTree()
//{
//    FILE *fp;
//	fp = fopen("c:\\kdtree.tree", "rb+");
//
//	// read binary data
//	size_t kdSize, triSize;
//	fread(&kdSize, sizeof(size_t), 1, fp);
//	KdNodeOrigin *h_kdNode = (KdNodeOrigin *)malloc(sizeof(KdNodeOrigin) * kdSize);
//	fread(h_kdNode, sizeof(KdNodeOrigin), kdSize, fp);
//
//	fread(&triSize, sizeof(size_t), 1, fp);
//	size_t *leafTriIndex = (size_t *)malloc(sizeof(size_t) * triSize);
//	fread(leafTriIndex, sizeof(size_t), triSize, fp);
//
//	// copy to GPU
//	kdNode_.size_ = kdSize;
//	CUDA_SAFE_CALL( cudaMemcpy( kdNode_.list_[0], h_kdNode, sizeof(KdNodeOrigin) * kdSize, cudaMemcpyHostToDevice ) );
//	leafPrims_.initList(10 * nPrims_);
//	leafPrims_.size_ = triSize;
//	CUDA_SAFE_CALL( cudaMemcpy( leafPrims_.list_[0], leafTriIndex, sizeof(size_t) * triSize, cudaMemcpyHostToDevice ) );
//
//	
//	
//	free(h_kdNode);
//	free(leafTriIndex);
//	fclose(fp);
//}

// kd tree构建
void GPU_BuildKdTree::buildTree()
{
	// cudaError_t e = cudaGetLastError();
	// printf("build tree error code: %d\n", e);

	// ******预热
	/*kdScan(	CUDPP_ADD, 
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
			NULL,
			NULL,
			0);*/


	//printf("prims: %d\n", nPrims_);

	//printf("building~~~~~\n");

	// measure time
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

    /*CuTimer buildTimer;
	buildTimer.startTimer();*/

	// 创建kd-tree的根节点
	createRoot();

	// 计算每个prim的包围盒
	calPrimsBoundingBox();

	// 
	int lHandleTime = 0;
	largeProcTime_ = 0;

	//CuTimer largeTimer;
	//largeTimer.startTimer();

	// 处理large node
	while (activeList_.size_ > 0)
	{
		isNeedFilter_ = true;
		
		
	    //CuTimer largeStepTimer;
		//largeStepTimer.startTimer();
		//TIMER_START(largeStepTimer);

		/*CuTimer proLNFunTimer;
		proLNFunTimer.startTimer();*/
		// handle large node 
		processLargeNode();
		//proLNFunTimer.finishTimer("///////processLargeNode function");

		/*CuTimer largeExtTimer;
		largeExtTimer.startTimer();*/
		// 把activeList添加到nodeList
		largeNodeAppendNodeList();

		// 把largeNodeList与active进行交换
		largeNodeSwap();

		// 内存扩展(动态表)
		largeNodeExtendMem();

		//largeExtTimer.finishTimer("~~~~~~large extent ");

		largeProcTime_++;
		lHandleTime++;
		//printf("handle time: %d, size: %d\n", lHandleTime, activeList_.size_);
		//largeStepTimer.finishTimer("******large node handle each step");
		//TIMER_FINISH(largeStepTimer, "*******large node handle each step");


	}

	
	
   // printf("large node handle over!\n");
	
	

	// 释放largeNode资源
	largeNodeRelease();
	//largeTimer.finishTimer("large node handle");
	

	//CuTimer smallTime;
	//smallTime.startTimer();

	// 对small node进行预处理
	preProcessSmallNode();

	// 初始化处理small node的一些数据, 在正式处理small node之前
	smallNode_initProcessSmallNode();

	//int smallHandleTime = 0;
	smallProcTime_ = 0;

	// 对small node进行正式处理
	while (smallNodeList_.size_ > 0)
	{
		//printf("smallSize: %d\n", smallNodeList_.size_);
		
		// handle small node
		processSmallNode();
		//printf("smallSize: %d\n", smallNodeList_.size_);
		
		// 把smallNodeList添加到nodeList
		smallNodeAppendNodeList();
		//printf("smallSize: %d\n", smallNodeList_.size_);

		// 把smallNodeList与smallNodeNextList进行交换
		smallNodeSwap();
		//printf("smallSize: %d\n", smallNodeList_.size_);

		// 对small node中的内存扩展(动态表)
		smallNodeExtendMem();


		//printf("smallhandletime: %d\n", ++smallHandleTime);
		//printf("smallSize: %d\n", smallNodeList_.size_);
		smallProcTime_++;

		

	}
	//printf("small node handle over!\n");


	// 释放smallNode资源
	smallNodeRelease();

	//smallTime.finishTimer("small node handle");

	//buildTimer.finishTimer("GPU kd-tree construction");
	cutStopTimer(timer);
	//printf("Build kd-tree takes: %f ms\n", cutGetTimerValue(timer));

	// printf("kaka\n");

	// e = cudaGetLastError();
	// printf("finish kd error code: %d\n", e);

	// save kdtree
	//saveKdTree();

	// kd node present
	/*char savepath[200] = "c:\\1.obj";
	saveBB(savepath);*/

	

	// print result
	// print kd nodes
	// print result
	// print kd nodes
	// ***
	/*FILE *fp;
	fp = fopen(resultPath_, "a");
	int vNum = nVertex_;

	int count = 0;
	int leafCount = 0;
	int emptyLeafCount = 0;
	int allLeafPrims = 0;
 	KdNodeOrigin *h_kdNode = (KdNodeOrigin *)malloc(sizeof(KdNodeOrigin) * kdNode_.size_);
	size_t *leafTriIndex = (size_t *)malloc(sizeof(size_t) * leafPrims_.size_);
	CUDA_SAFE_CALL(cudaMemcpy(leafTriIndex, leafPrims_.list_[0], sizeof(size_t) * leafPrims_.size_, cudaMemcpyDeviceToHost));
	int nodeListSize = nodeList_.size();
    vector<int> leafAddr;

	CUDA_SAFE_CALL(cudaMemcpy(h_kdNode, kdNode_.list_[0], sizeof(KdNodeOrigin) * kdNode_.size_, cudaMemcpyDeviceToHost));

	int testAddr = 0;*/

	//printf("kd node:\n");
	//for (int i = 0; i < nodeListSize; i++)
	//{
	//    int *nodeList = nodeList_[i].list_;
	//	int size = nodeList_[i].size_;
	//	int *h_nodeList = (int *)malloc(sizeof(int) * size);
	//	CUDA_SAFE_CALL(cudaMemcpy(h_nodeList, nodeList, sizeof(int) * size, cudaMemcpyDeviceToHost));
	//	for (int j = 0; j < size; j++)
	//	{
	//	    //CUDA_SAFE_CALL(cudaMemcpy(h_kdNode + count, kdNode_.list_[0] + h_nodeList[j], sizeof(KdNodeOrigin), cudaMemcpyDeviceToHost));
	//		
	//		// bounding box
	//		// write vertex
	//		if (4 == h_kdNode[count].splitType_)
	//		{
	//			
	//			//printf("%d ", h_kdNode[count].nTris_);
	//			float x[2], y[2], z[2];
	//			x[0] = h_kdNode[count].bb_.bbMin_.x;
	//			x[1] = h_kdNode[count].bb_.bbMax_.x;
	//			y[0] = h_kdNode[count].bb_.bbMin_.y;
	//			y[1] = h_kdNode[count].bb_.bbMax_.y;
	//			z[0] = h_kdNode[count].bb_.bbMin_.z;
	//			z[1] = h_kdNode[count].bb_.bbMax_.z;

	//			// BB points
	//			/*for (int i = 0; i < 2; i++)
	//			{
	//				for (int j = 0; j < 2; j++)
	//				{
	//					for (int k = 0; k < 2; k++)
	//					{
	//					   fprintf(fp, "v %f %f %f\n", x[i], y[j], z[k]); 
	//					}
	//				}
	//			}*/

	//			// face model
	//			/*int vertexBox[6][4] = {1, 2, 4, 3, 
	//								1, 5, 6, 2,
	//								6, 5, 7, 8,
	//								7, 8, 4, 3, 
	//								2, 4, 8, 6,
	//								3, 7, 5, 1};
	//			for (int i = 0; i < 6; i++)
	//			{
	//				for (int j = 0; j < 4; j++)
	//				{
	//					vertexBox[i][j] += vNum;
	//				}
	//			}
	//			for (int i = 0; i < 6; i++)
	//				fprintf(fp, "f %d %d %d %d\n", vertexBox[i][0], vertexBox[i][1],
	//				vertexBox[i][2], vertexBox[i][3]);*/

	//			// line model
	//			/*int vertexBox[12][3] = {1, 1, 2, 
	//								1, 1, 3,
	//								3, 3, 4,
	//								4, 4, 2,
	//								2, 2, 6,
	//								4, 4, 8,
	//								1, 1, 5,
	//								3, 3, 7,
	//								8, 8, 6,
	//								6, 6, 5,
	//								5, 5, 7,
	//								7, 7, 8
	//							};
	//			for (int i = 0; i < 12; i++)
	//			{
	//				for (int j = 0; j < 3; j++)
	//				{
	//					vertexBox[i][j] += vNum;
	//				}
	//			}
	//			for (int i = 0; i < 12; i++)
	//				fprintf(fp, "f %d %d %d\n", vertexBox[i][0], vertexBox[i][1],
	//				vertexBox[i][2]);*/


	//			//vNum += 8;

	//			size_t addr = h_kdNode[count].splitValue_;

	//			// save leaf tris index addr
	//			leafAddr.push_back(addr);

	//			// write leaf index
	//			leafCount ++;
	//		
	//		
	//		    int triNum = h_kdNode[count].nTris_;
	//			allLeafPrims += triNum;
	//			if (0 == triNum) emptyLeafCount++;
	//			for (int j = 0; j < triNum; j++)
	//			{
	//				int triIndex = leafTriIndex[addr + j];
	//				uint4 vertexIndex = prims_.h_face_[triIndex];
	//			    fprintf(fp, "f %d %d %d\n", vertexIndex.x + 1, vertexIndex.y + 1, vertexIndex.z + 1);
	//			}
	//			

	//		}

	//		// writting all bounding box
	//		float x[2], y[2], z[2];
	//		x[0] = h_kdNode[count].bb_.bbMin_.x;
	//		x[1] = h_kdNode[count].bb_.bbMax_.x;
	//		y[0] = h_kdNode[count].bb_.bbMin_.y;
	//		y[1] = h_kdNode[count].bb_.bbMax_.y;
	//		z[0] = h_kdNode[count].bb_.bbMin_.z;
	//		z[1] = h_kdNode[count].bb_.bbMax_.z;

	//		// BB points
	//		/*for (int i = 0; i < 2; i++)
	//		{
	//			for (int j = 0; j < 2; j++)
	//			{
	//				for (int k = 0; k < 2; k++)
	//				{
	//				   fprintf(fp, "v %f %f %f\n", x[i], y[j], z[k]); 
	//				}
	//			}
	//		}*/

	//		// face model
	//		/*int vertexBox[6][4] = {1, 2, 4, 3, 
	//							1, 5, 6, 2,
	//							6, 5, 7, 8,
	//							7, 8, 4, 3, 
	//							2, 4, 8, 6,
	//							3, 7, 5, 1};
	//		for (int i = 0; i < 6; i++)
	//		{
	//			for (int j = 0; j < 4; j++)
	//			{
	//				vertexBox[i][j] += vNum;
	//			}
	//		}
	//		for (int i = 0; i < 6; i++)
	//			fprintf(fp, "f %d %d %d %d\n", vertexBox[i][0], vertexBox[i][1],
	//			vertexBox[i][2], vertexBox[i][3]);*/

 //           //vNum += 8;
	//		// wirte all bounding box

	//		count ++;

	//	}
	//}

	// bounding box
	//float x[2], y[2], z[2];
	//x[0] = h_kdNode[0].bb_.bbMin_.x;
	//x[1] = h_kdNode[0].bb_.bbMax_.x;
	//y[0] = h_kdNode[0].bb_.bbMin_.y;
	//y[1] = h_kdNode[0].bb_.bbMax_.y;
	//z[0] = h_kdNode[0].bb_.bbMin_.z;
	//z[1] = h_kdNode[0].bb_.bbMax_.z;

	//// BB points
	//for (int i = 0; i < 2; i++)
	//{
	//	for (int j = 0; j < 2; j++)
	//	{
	//		for (int k = 0; k < 2; k++)
	//		{
	//		   fprintf(fp, "v %f %f %f\n", x[i], y[j], z[k]); 
	//		}
	//	}
	//}

	//// face model
	//int vertexBox[6][4] = {1, 2, 4, 3, 
	//					1, 5, 6, 2,
	//					6, 5, 7, 8,
	//					7, 8, 4, 3, 
	//					2, 4, 8, 6,
	//					3, 7, 5, 1};
	//for (int i = 0; i < 6; i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		vertexBox[i][j] += vNum;
	//	}
	//}
	//for (int i = 0; i < 6; i++)
	//	fprintf(fp, "f %d %d %d %d\n", vertexBox[i][0], vertexBox[i][1],
	//	vertexBox[i][2], vertexBox[i][3]);
	//vNum += 8;


	// DFS kd-tree
	//stack<int> S;
 //   int index = 0;
	//while(1)
	//{
	//	while(index != -1)
	//	{
	//	    if (h_kdNode[index].splitType_ != 4)
	//		{
	//			S.push(h_kdNode[index].rChild_);
	//		}
	//		else
	//		{
	//			size_t addr = h_kdNode[index].splitValue_;

	//			// save leaf tris index addr
	//			//leafAddr.push_back(addr);

	//			// write leaf index
	//			leafCount ++;
	//		
	//		
	//		    int triNum = h_kdNode[index].nTris_;
	//			allLeafPrims += triNum;
	//			if (0 == triNum) emptyLeafCount++;
	//			for (int j = 0; j < triNum; j++)
	//			{
	//				int triIndex = leafTriIndex[addr + j];
	//				uint4 vertexIndex = prims_.h_face_[triIndex];
	//			    fprintf(fp, "f %d %d %d\n", vertexIndex.x + 1, vertexIndex.y + 1, vertexIndex.z + 1);
	//			}
	//		}
	//		index = h_kdNode[index].lChild_;
	//	}
	//	if (S.empty()) break;
	//	index = S.top();
	//	S.pop();
	//}

	// ray tracing test

	//int rayNum = 100000;
	//sendRayTest(rayNum);
	//cudaError_t eray;
	//eray = cudaGetLastError();

	//float* h_tHit;
	//float4* h_dir;
	//int* h_hitTri;
	//h_tHit = (float *)malloc(sizeof(float) * rayNum);
	//CUDA_SAFE_CALL(cudaMemcpy(h_tHit, d_tHit, sizeof(float) * rayNum, cudaMemcpyDeviceToHost));
	//h_dir = (float4 *)malloc(sizeof(float4) * rayNum);
	//CUDA_SAFE_CALL(cudaMemcpy(h_dir, d_dir, sizeof(float4) * rayNum, cudaMemcpyDeviceToHost));
	//h_hitTri = (int *)malloc(sizeof(size_t) * rayNum);
	//CUDA_SAFE_CALL(cudaMemcpy(h_hitTri, d_hitTri, sizeof(int) * rayNum, cudaMemcpyDeviceToHost));
	//
	//// src
	//int srcpos;
	//int infinite = 3;
	//float4 src;
	//src.x = 0.3f;
	//src.y = 0.3f;
	//src.z = 0.3f;
	//srcpos = vNum + 1;
	//fprintf(fp, "v %f %f %f\n", src.x, src.y, src.z);
	//fprintf(fp, "v %f %f %f\n", 0.0f, 0.0f, 0.0f);
	//vNum += 2;
	//fprintf(fp, "f %d %d %d\n", vNum-1, vNum-1, vNum-1);
	//

	// print rand num
	//size_t* h_rand = (size_t *)malloc(sizeof(size_t) * rayNum * 2);
	//CUDA_SAFE_CALL(cudaMemcpy(h_rand, d_rand, sizeof(size_t) * rayNum * 2, cudaMemcpyDeviceToHost));
	/*printf("rand:\n");
	for (int i = 0; i < 2 * rayNum; i++)
	{
		printf("%u ", h_rand[i] % 100);
	}
	printf("\n");*/

	// ray
	/*for (int i = 0; i < rayNum; i++)
	{
	    if (h_tHit[i] > 0 && h_hitTri[i] > 0)
		{
			fprintf(fp, "v %f %f %f\n", src.x + h_tHit[i] * h_dir[i].x, 
									src.y + h_tHit[i] * h_dir[i].y, 
									src.z + h_tHit[i] * h_dir[i].z);
			fprintf(fp, "f %d %d %d\n", srcpos, vNum + 1, srcpos);

			vNum++;
		}*/

		/*if (h_hitTri[i] == -2)
		{
		    fprintf(fp, "v %f %f %f\n", src.x + 1.0 * h_dir[i].x, 
									src.y + 1.0 * h_dir[i].y, 
									src.z + 1.0 * h_dir[i].z);
			fprintf(fp, "f %d %d %d\n", srcpos, vNum + 1, srcpos);

			vNum++;
		}*/
		/*else
		{
			fprintf(fp, "v %f %f %f\n", src.x + infinite * h_dir[i].x, 
									src.y + infinite * h_dir[i].y, 
									src.z + infinite * h_dir[i].z);
			fprintf(fp, "f %d %d %d\n", srcpos, vNum + 1, srcpos);
		}*/
		
		// dir
		/*printf("%f %f %f %f\n", h_dir[i].x, h_dir[i].y, h_dir[i].z, 
			h_dir[i].x * h_dir[i].x + h_dir[i].y * h_dir[i].y + h_dir[i].z * h_dir[i].z);*/
		// hit
		//if (h_tHit[i] > 0.1f) printf("%f\n", h_tHit[i]);

		// hit tri
		//if (h_hitTri[i] > 0) 
		//{
		//	printf("%d\n", h_hitTri[i]);
		//	fprintf(fp, "v %f %f %f\n", src.x + 4 * h_dir[i].x, 
		//							src.y + 4 * h_dir[i].y, 
		//							src.z + 4 * h_dir[i].z);
		//	fprintf(fp, "f %d %d %d\n", srcpos, vNum + 1, srcpos);

		//	vNum ++;

		//	/*uint4 vertexIndex = prims_.h_face_[h_hitTri[i]];
		//	fprintf(fp, "f %d %d %d\n", vertexIndex.x + 1, vertexIndex.y + 1, vertexIndex.z + 1);*/
		//}
	//}


	/*fclose(fp);*/

	//printf("kd nodes: %d\n", count);
	//printf("leaf nodes: %d\n", leafCount);
	//printf("empty leaf nodes: %d\n", emptyLeafCount);
	//printf("all leaf prims: %d\n", allLeafPrims);


	//printf("leaf tris index addr:\n");
	//for (int i = 0; i < leafAddr.size(); i++)
	//{
	//    printf("%d ", leafAddr[i]);
	//}
	//printf("\n");

	//printf("leafNum: %u\n", leafNum_);

	//// print leafprims
	//size_t *h_leafPrims = (size_t *)malloc(sizeof(size_t) * leafPrims_.size_);
	//CUDA_SAFE_CALL(cudaMemcpy(h_leafPrims, leafPrims_.list_[0], sizeof(size_t) * leafPrims_.size_, cudaMemcpyDeviceToHost));
	//printf("leafPrims:\n");
	//for (int i = 0; i < leafPrims_.size_; i++)
	//{
	//	if (i < leafPrims_.size_ - 1)
	//	{
	//		printf("%u, ", h_leafPrims[i]);
	//	}
	//	else
	//	{
	//	    printf("%u", h_leafPrims[i]);
	//	}
	//	
	//}
	//printf("\n");

	//printf("Kd-Tree building over!\n");

	// 
	//rayTrace();
}

//void GPU_BuildKdTree::saveBB(char* path)
//{
//	int vNum = 0;
//	// file
//	FILE *fp;
//	fp = fopen(path, "w+");
//
//	printf("%d\n", kdNodeBase_.size_);
//
//	// host mem
//	KdNode_base* h_base = (KdNode_base*)malloc(sizeof(KdNode_base) * kdNodeBase_.size_); 
//	KdNode_bb* h_bb = (KdNode_bb*)malloc(sizeof(KdNode_bb) * kdNodeBase_.size_);
//	KdNode_extra* h_extra = (KdNode_extra*)malloc(sizeof(KdNode_extra) * kdNodeBase_.size_);
//	size_t *leafTriIndex = (size_t *)malloc(sizeof(size_t) * leafPrims_.size_);
//
//	
//	// copy
//	cudaMemcpy(h_base, kdNodeBase_.list_[0], sizeof(KdNode_base) * kdNodeBase_.size_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_bb, kdNodeBB_.list_[0], sizeof(KdNode_bb) * kdNodeBase_.size_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_extra, kdNodeExtra_.list_[0], sizeof(KdNode_extra) * kdNodeBase_.size_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(leafTriIndex, leafPrims_.list_[0], sizeof(size_t) * leafPrims_.size_, cudaMemcpyDeviceToHost);
//
//	/*printf("leafTriIndex\n");
//	for (int i = 0; i < 100; i++)
//	{
//		printf("%d, ", leafTriIndex[i]);
//	}*/
//
//	// write all tris
//	for (int i = 0; i < prims_.vertexNum_; i++)
//	{
//	    fprintf(fp, "v %f %f %f\n", prims_.h_vertex_[i].x, 
//			prims_.h_vertex_[i].y, prims_.h_vertex_[i].z);
//	}
//	vNum = prims_.vertexNum_;
//
//	// write all face
//	/*for (int i = 0; i < prims_.faceNum_; i++)
//	{
//	    fprintf(fp, "f %d %d %d\n", prims_.h_face_[i].x + 1, 
//			prims_.h_face_[i].y + 1, prims_.h_face_[i].z + 1);
//	}*/
//
//
//	// DFS kd-tree
//	stack<int> S;
//    int index = 0;
//
//	int leafCount = 0;
//	while(1)
//	{
//		while(index != -1)
//		{
//			// bb
//			float x[2], y[2], z[2];
//			x[0] = h_bb[index].bb_.bbMin_.x;
//			x[1] = h_bb[index].bb_.bbMax_.x;
//			y[0] = h_bb[index].bb_.bbMin_.y;
//			y[1] = h_bb[index].bb_.bbMax_.y;
//			z[0] = h_bb[index].bb_.bbMin_.z;
//			z[1] = h_bb[index].bb_.bbMax_.z;
//
//			// BB points
//			for (int i = 0; i < 2; i++)
//			{
//				for (int j = 0; j < 2; j++)
//				{
//					for (int k = 0; k < 2; k++)
//					{
//					   fprintf(fp, "v %f %f %f\n", x[i], y[j], z[k]); 
//					}
//				}
//			}
//
//			// face model
//			int vertexBox[6][4] = {1, 2, 4, 3, 
//								1, 5, 6, 2,
//								6, 5, 7, 8,
//								7, 8, 4, 3, 
//								2, 4, 8, 6,
//								3, 7, 5, 1};
//			for (int i = 0; i < 6; i++)
//			{
//				for (int j = 0; j < 4; j++)
//				{
//					vertexBox[i][j] += vNum;
//				}
//			}
//			for (int i = 0; i < 6; i++)
//				fprintf(fp, "f %d %d %d %d\n", vertexBox[i][0], vertexBox[i][1],
//				vertexBox[i][2], vertexBox[i][3]);
//
//			vNum += 8;
//
//
//		    if (h_base[index].splitType_ != 4)
//			{
//				S.push(h_base[index].rChild_);
//			}
//			else
//			{
//				size_t addr = *((size_t *)&h_base[index].splitValue_);
//
//				// save leaf tris index addr
//				//leafAddr.push_back(addr);
//
//				// write leaf index
//				leafCount ++;
//			
//			
//			    int triNum = h_extra[index].nTris_;
//				printf("%d, ", triNum);
//				//allLeafPrims += triNum;
//				//if (0 == triNum) emptyLeafCount++;
//				for (int j = 0; j < triNum; j++)
//				{
//					
//					int triIndex = leafTriIndex[addr + j];
//					//printf("addr: %d, triIdx: %d\n", addr, triIndex);
//					int4 vertexIndex = prims_.h_face_[triIndex];
//				    fprintf(fp, "f %d %d %d\n", vertexIndex.x + 1, vertexIndex.y + 1, vertexIndex.z + 1);
//				}
//			}
//			index = h_base[index].lChild_;
//		}
//		if (S.empty()) break;
//		index = S.top();
//		S.pop();
//	}
//
//
//	// release
//	free(h_base);
//	free(h_bb);
//	free(h_extra);
//	free(leafTriIndex);
//
//	fclose(fp);
//}

// 把largeNodeList与active进行交换
void GPU_BuildKdTree::largeNodeSwap()
{
	//~kdNode_.size_ += nextList_.size_;
	kdNodeBase_.size_ += nextList_.size_;
	
	if (isNeedFilter_) // 需要进行filter
	{
		activeList_ = largeNodeNextList_;

		swapList<size_t, 2>(triNodeList_, largeNodeTriNodeList_);

		// alloc mem for largeNodeNextList
		largeNodeNextList_.initList(2 * activeList_.size_);
	}
	//%
	else
	{
		activeList_ = nextList_;
		swapList<size_t, 2>(triNodeList_, newTriNodeList_);

		// alloc
		nextList_.initList(2 * activeList_.size_);
	}
	
	
    

}

// 对largeNode处置中的一些内存进行扩展(动态表)
void GPU_BuildKdTree::largeNodeExtendMem()
{
    // kdNode_ 
	//~kdNode_.realloc(kdNode_.size_ + 2 * activeList_.size_, true);
	kdNodeBase_.realloc(kdNodeBase_.size_ + 2 * activeList_.size_, true);
	kdNodeBB_.realloc(kdNodeBase_.size_ + 2 * activeList_.size_, true);
	kdNodeExtra_.realloc(kdNodeBase_.size_ + 2 * activeList_.size_, true);
	
	// activeList_

	// nextList_
	//%
	if (isNeedFilter_)
	{
		nextList_.realloc(2 * activeList_.size_, false);
	}
	else
	{
		largeNodeNextList_.realloc(2 * activeList_.size_, false);
	}
	
	

	// newTriNodeList_
	newTriNodeList_.realloc(2 * triNodeList_.size_, false);

	// largeNodeTriNodeList_
	//largeNodeTriNodeList_.realloc(triNodeList_.size_, false);
	largeNodeTriNodeList_.realloc(2 * triNodeList_.size_, false);

	// smallNodeNextList_
	smallNodeNextList_.realloc(smallNodeNextListSizeGlobal_ + 2 * activeList_.size_, true);

	// smallNodeTriNodeList_
	//smallNodeTriNodeList_.realloc(smallNodeTriNodeListSizeGlobal_ + triNodeList_.size_, true);
	smallNodeTriNodeList_.realloc(smallNodeTriNodeListSizeGlobal_ + 2 * triNodeList_.size_, true);

	// triNodePrimsBBListAssist_
	d_itriNodePrimsBBListAssist_.realloc(triNodeList_.size_, false);

	d_otriNodePrimsBBListAssist_.realloc(triNodeList_.size_, false);

	// splitListAssist_
	d_isplitListAssist_.realloc(triNodeList_.size_, false);
	d_osplitListAssist_.realloc(triNodeList_.size_, false);
	
	// splitTypeListAssist_
	d_isplitTypeListAssist_.realloc(triNodeList_.size_, false);
	d_osplitTypeListAssist_.realloc(triNodeList_.size_, false);

    // leftChildListFlagsAssist_
	d_ileftChildListFlagsAssist_.realloc(triNodeList_.size_, false);
	d_oleftChildListFlagsAssist_.realloc(triNodeList_.size_, false);

	// rightChildListFlagsAssist_
	d_irightChildListFlagsAssist_.realloc(triNodeList_.size_, false);
	d_orightChildListFlagsAssist_.realloc(triNodeList_.size_, false);

    // nextNodeNumListAssist_
	d_inextNodeNumListAssist_.realloc(2 * activeList_.size_, false);
	d_onextNodeNumListAssist_.realloc(2 * activeList_.size_, false);

	// largeNodeFlagsListAssist_
	d_ilargeNodeFlagsListAssist_.realloc(2 * activeList_.size_, false);
	d_olargeNodeFlagsListAssist_.realloc(2 * activeList_.size_, false);
	
	// largeNodeNumListAssist_
	d_ilargeNodeNumListAssist_.realloc(2 * activeList_.size_, false);
	d_olargeNodeNumListAssist_.realloc(2 * activeList_.size_, false);
	
	// largeNodeTriNodeFlagsListAssist_
	d_ilargeNodeTriNodeFlagsListAssist_.realloc(2 * triNodeList_.size_, false);
	d_olargeNodeTriNodeFlagsListAssist_.realloc(2 * triNodeList_.size_, false);
 
    // smallNodeFlagsListAssist_
	d_ismallNodeFlagsListAssist_.realloc(2 * activeList_.size_, false);
	d_osmallNodeFlagsListAssist_.realloc(2 * activeList_.size_, false);
	
	// smallNodeNumListAssist_
	d_ismallNodeNumListAssist_.realloc(2 * activeList_.size_, false);
    d_osmallNodeNumListAssist_.realloc(2 * activeList_.size_, false);
	
	// smallNodeTriNodeFlagsListAssist_
	d_ismallNodeTriNodeFlagsListAssist_.realloc(2 * triNodeList_.size_, false);
	d_osmallNodeTriNodeFlagsListAssist_.realloc(2 * triNodeList_.size_, false);

}
template <typename T, int dimension>
inline void GPU_BuildKdTree::swapList(List<T, dimension> &list1, List<T, dimension> &list2)
{
    T *listTemp[dimension];
	size_t sizeTemp;
	size_t capacityTemp;
	
	for (int i = 0; i < dimension; i++)
	{
	    listTemp[i] = list1.list_[i];
	}
	sizeTemp = list1.size_;
	capacityTemp = list1.capacity_;

	for (int i = 0; i < dimension; i++)
	{
	    list1.list_[i] = list2.list_[i];
	}
	list1.size_ = list2.size_;
	list1.capacity_ = list2.capacity_;

	for (int i = 0; i < dimension; i++)
	{
	    list2.list_[i] = listTemp[i];
	}
	list2.size_ = sizeTemp;
	list2.capacity_ = capacityTemp; 
}

// 把smallNodeList与smallNodeNextList进行交换
void GPU_BuildKdTree::smallNodeSwap()
{
	smallNodeList_ = smallNodeNextList_;

	swapList<size_t, 1>(smallNodeRList_, smallNodeNextListRList_);


	smallNodeNextListRList_.size_ = 0;

	swapList<size_t, 1>(smallNodeMaskHi_, smallNodeNextListMaskHi_);

	swapList<size_t, 1>(smallNodeMaskLo_, smallNodeNextListMaskLo_);


	// alloc mem for smallNodeNextList
	smallNodeNextList_.initList(2 * smallNodeNextList_.size_);
}

// 对smallNode处置中的一些内存进行扩展(动态表)
void GPU_BuildKdTree::smallNodeExtendMem()
{
	// kd node extend
	//~kdNode_.realloc(kdNode_.size_ + 2 * smallNodeList_.size_, true);
	kdNodeBase_.realloc(kdNodeBase_.size_ + 2 * smallNodeList_.size_, true);
	kdNodeBB_.realloc(kdNodeBase_.size_ + 2 * smallNodeList_.size_, true);
	kdNodeExtra_.realloc(kdNodeBase_.size_ + 2 * smallNodeList_.size_, true);

	smallNodeNextListMaskHi_.realloc(2 * smallNodeList_.size_, false);
	smallNodeNextListMaskLo_.realloc(2 * smallNodeList_.size_, false);
	
	smallNodeNextListRList_.realloc(2 * smallNodeList_.size_, false);
	
	d_ismallNodeLeafFlags_.realloc(smallNodeList_.size_, false);
	d_osmallNodeLeafFlags_.realloc(smallNodeList_.size_, false);
	d_ismallNodeNoLeafFlags_.realloc(smallNodeList_.size_, false);
    d_osmallNodeNoLeafFlags_.realloc(smallNodeList_.size_, false);
	d_ismallNodeLeafSize_.realloc(smallNodeList_.size_, false);
	d_osmallNodeLeafSize_.realloc(smallNodeList_.size_, false);

	// problem****************************************************
	leafPrims_.realloc(leafPrims_.size_ + nPrims_, true);

}


// 把activeList添加到nodeList中
void GPU_BuildKdTree::largeNodeAppendNodeList()
{
	Node newNode;
	newNode.list_ = activeList_.list_[0];
	newNode.size_ = activeList_.size_;

	nodeList_.push_back(newNode);
}

// 把smallNodeList添加到nodeList中
void GPU_BuildKdTree::smallNodeAppendNodeList()
{
	Node newNode;
	newNode.list_ = smallNodeList_.list_[0];
	newNode.size_ = smallNodeList_.size_;

	nodeList_.push_back(newNode);
}


// 创建kd-tree根节点
void GPU_BuildKdTree::createRoot()
{
	size_t everyBlockNum = CREATE_ROOT_BLOCK_SIZE * CREATE_ROOT_THREAD_HANDLE_NUM;
	size_t gridSize = (nPrims_ & (everyBlockNum - 1)) ? nPrims_ / everyBlockNum + 1 : nPrims_ / everyBlockNum;
	size_t blockSize = CREATE_ROOT_BLOCK_SIZE;

	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// 产生一个新的activeList
	activeList_.initList(1);


	activeList_.size_ = 1;
	triNodeList_.size_ = nPrims_;
	//~kdNode_.size_ = 1;
	kdNodeBase_.size_ = 1;

	// 设置segflags
	CUDA_SAFE_CALL(cudaMemset(triNodeList_.list_[1], 0, sizeof(size_t) * triNodeList_.size_));

	// run kernal
	if (gridSize > 0)
	{
	    createRootKernal<<< gDim, bDim>>>(   nPrims_,
												activeList_.list_[0],
												//~kdNode_.list_[0],
												kdNodeBase_.list_[0],
												kdNodeBB_.list_[0],
												kdNodeExtra_.list_[0],
												triNodeList_.list_[0],
												triNodeList_.list_[1]);
	}
	

}

// 计算每个bezier曲面的包围盒
void GPU_BuildKdTree::calPrimsBoundingBox()
{
	size_t blockSize = CAL_PRIMS_BB_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * CAL_PRIMS_BB_THREAD_HANDLE_NUM;
	size_t gridSize = (nPrims_ & (everyBlockNum - 1)) ? nPrims_ / everyBlockNum + 1 : nPrims_ / everyBlockNum;

	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// run kernal
	if (gridSize > 0)
	{
	    calPrimsBoundingBoxKernal<<< gDim, bDim>>>( nPrims_,
													allBoundingBoxList_.list_[0],
													allBoundingBoxList_.list_[1],
													prims_.d_vertex_[keyframe_],
													prims_.d_face_[keyframe_]
													);
	}
	

}

// 收集triNodeList中所有node的BB,存入triNodePrimsBBList中去，
void GPU_BuildKdTree::collectTriNodePrimsBB()
{
	size_t blockSize = COLLECT_TRINODE_BB_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * COLLECT_TRINODE_BB_THREAD_HANDLE_NUM;

	// ***得到tri-node的大小，可估计
	size_t gridSize = (triNodeList_.size_ & (everyBlockNum - 1)) ? triNodeList_.size_ / everyBlockNum + 1 : triNodeList_.size_ / everyBlockNum;

	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// 收集tri-node 中的所有的tri的bb
	if (gridSize > 0)
	{
	    collectTriNodePrimsBBKernal<<<gDim, bDim, 0>>>( 
														allBoundingBoxList_.list_[0],
														allBoundingBoxList_.list_[1],
														d_itriNodePrimsBBListAssist_.list_[0],
														d_itriNodePrimsBBListAssist_.list_[1],
														d_itriNodePrimsBBListAssist_.list_[2],
														d_itriNodePrimsBBListAssist_.list_[3],
														d_itriNodePrimsBBListAssist_.list_[4],
														d_itriNodePrimsBBListAssist_.list_[5],


														triNodeList_.list_[0],
														triNodeList_.size_);
	}
	
}

// 计算activeList中每个node的包围盒,以d_itriNodePrimsBBListAssist_作为data,
// triNodeList:segFlags作为flags, 执行segmented scan
void GPU_BuildKdTree::calActiveNodeBoundingBox()
{
	// set argument 
	if (activeList_.size_ > 1)
	{
		for (size_t i = 0; i < 6; i++)
		{
			CUDPPOperator op = (i < 3) ? CUDPP_MIN : CUDPP_MAX;
			kdSegScan(	op, 
				CUDPP_FLOAT, 
				CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE, 
				d_otriNodePrimsBBListAssist_.list_[i], 
				d_itriNodePrimsBBListAssist_.list_[i], 
				triNodeList_.list_[1],
				triNodeList_.size_);
		}
	}
	else
	{
		for (size_t i = 0; i < 6; i++)
		{
			CUDPPOperator op = (i < 3) ? CUDPP_MIN : CUDPP_MAX;
			kdScan(	op, 
				CUDPP_FLOAT, 
				CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE, 
				d_otriNodePrimsBBListAssist_.list_[i], 
				d_itriNodePrimsBBListAssist_.list_[i], 
				triNodeList_.size_);
		}
	}
	



}

// 把activeNodeList中所有node的splitvalue, splittype，分配到辅助list中
// 执行segmented scan
void GPU_BuildKdTree::distributeSplit()
{
	//// splitList的segmented scan
	kdSegScan(	CUDPP_MAX, 
		CUDPP_FLOAT, 
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE, 
		d_osplitListAssist_.list_[0],
		d_isplitListAssist_.list_[0], 
		triNodeList_.list_[1],
		triNodeList_.size_);

	//// splitList的segmented scan
	kdSegScan(	CUDPP_MAX, 
		CUDPP_UINT, 
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE, 
		d_osplitTypeListAssist_.list_[0],
		d_isplitTypeListAssist_.list_[0], 
		triNodeList_.list_[1],
		triNodeList_.size_);
}


// 设置activeNodeList中所有node的boundingbox, splittype, splitValue,为distribute做准备
void GPU_BuildKdTree::set_ActiveNode_Child_PreDistribute()
{
	//// 在GPU上进行设置每个activeNode中node的boundingbox



	// init splitList
	size_t blockSize = SET_DEVICE_MEM_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * SET_DEVICE_MEM_THREAD_HANDLE_NUM;
	size_t gridSize = (triNodeList_.size_ & (everyBlockNum - 1)) ? triNodeList_.size_ / everyBlockNum + 1 : triNodeList_.size_ / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	//printf("triNodeList size: %d\n", triNodeList_.size_);

	if (gridSize > 0)
	{
	    deviceMemsetKernal<float><<<gDim, bDim, 0>>>(d_isplitListAssist_.list_[0], FLOAT_MIN, triNodeList_.size_);
	}
	

	// init splitTypeList
	CUDA_SAFE_CALL(cudaMemset(d_isplitTypeListAssist_.list_[0], 0, triNodeList_.size_ * sizeof(size_t)));

	blockSize = SET_ACTIVE_NODE_BB_BLOCK_SIZE;
	everyBlockNum = blockSize * SET_ACTIVE_NODE_BB_THREAD_HANDLE_NUM;
	nextList_.size_ = 2 * activeList_.size_;
	gridSize = (activeList_.size_ & (everyBlockNum - 1)) ? activeList_.size_ / everyBlockNum + 1 : activeList_.size_ / everyBlockNum;

	bDim.x = blockSize;
	gDim.x = gridSize;

	//// 设置activeNodeList中所有node的boundingbox, splittype, splitValue
	//// 并设置左右孩子节点的father, flag
	if (gridSize > 0)
	{
	    set_ActiveNode_Child_PreDistribute_Kernal<<<gDim, bDim, 0>>>(	//d_otriNodePrimsBBListAssist_,
																			d_otriNodePrimsBBListAssist_.list_[0],
																			d_otriNodePrimsBBListAssist_.list_[1],
																			d_otriNodePrimsBBListAssist_.list_[2],
																			d_otriNodePrimsBBListAssist_.list_[3],
																			d_otriNodePrimsBBListAssist_.list_[4],
																			d_otriNodePrimsBBListAssist_.list_[5],

																			activeList_.list_[0],
																			nextList_.list_[0],
																			//~kdNode_.list_[0],
																			kdNodeBase_.list_[0],
																			kdNodeBB_.list_[0],
																			kdNodeExtra_.list_[0],
																			//~kdNode_.size_,
																			kdNodeBase_.size_,
																			activeList_.size_,
																			d_isplitListAssist_.list_[0],
																			isNeedTight_,
																			d_isplitTypeListAssist_.list_[0]);
	}
	
}
// 计算将来nextList中每个node中含有的prim的数目，已经

// 以split设置tri-node中所有tri归属到左孩子还是右孩子
void GPU_BuildKdTree::setFlagWithSplit()
{
	size_t blockSize = SET_FLAG_WITH_SPLIT_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * SET_FLAG_WITH_SPLIT_THREAD_HANDLE_NUM;

	// ***得到tri-node的大小，可估计
	size_t gridSize = (triNodeList_.size_ & (everyBlockNum - 1)) ? triNodeList_.size_ / everyBlockNum + 1 : triNodeList_.size_ / everyBlockNum;

	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// run kernal
	if (gridSize > 0)
	{
	    setFlagWithSplitKernal<<<gDim, bDim, 0>>>(	//allBoundingBoxList_.primsBoundingBox_,
													allBoundingBoxList_.list_[0],
													allBoundingBoxList_.list_[1],

													triNodeList_.list_[0],
													d_ileftChildListFlagsAssist_.list_[0],
													d_irightChildListFlagsAssist_.list_[0],
													d_osplitListAssist_.list_[0],
													d_osplitTypeListAssist_.list_[0],
													triNodeList_.size_);
	}
    
	
}

// 计算将来nextList中每个node中含有prim的数目
void GPU_BuildKdTree::calNextListNodePrimNum()
{
	//// left child list flags segment scan
	kdSegScan(	CUDPP_ADD, 
		CUDPP_UINT, 
		CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE, 
		d_oleftChildListFlagsAssist_.list_[0],
		d_ileftChildListFlagsAssist_.list_[0], 
		triNodeList_.list_[1],
		triNodeList_.size_);

	//cudaError_t e1 = cudaGetLastError();

	//// right child list flags segment scan
	kdSegScan(	CUDPP_ADD, 
		CUDPP_UINT, 
		CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE, 
		d_orightChildListFlagsAssist_.list_[0],
		d_irightChildListFlagsAssist_.list_[0], 
		triNodeList_.list_[1],
		triNodeList_.size_);
}


// 把left/rightChildListFlags中的代表next list node中prims数目的值提出出来放入nextNodeNumList
void GPU_BuildKdTree::extractPrimNumFromChildFlags()
{
	// test
    /*if (triNodeList_.size_ > triNodeList_.capacity_)
		printf("size > capacity\n");*/

	//// compact left child node prims's num
	kdCompact(	CUDPP_UINT, 
		d_inextNodeNumListAssist_.list_[0], 
		d_oleftChildListFlagsAssist_.list_[0],
		triNodeList_.list_[1], 
		triNodeList_.size_);



	//// compact right child node prims's num
	kdCompact(	CUDPP_UINT, 
		d_inextNodeNumListAssist_.list_[0] + activeList_.size_, 
		d_orightChildListFlagsAssist_.list_[0],
		triNodeList_.list_[1], 
		triNodeList_.size_);
}

// 对nextNodeNumList进行exclusive scan，得到nextNodeListNew在newTriNodeList中的起始地址
void GPU_BuildKdTree::calNewTriNodeListAddr()
{
	
	
    kdScan(	CUDPP_ADD, 
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
			d_onextNodeNumListAssist_.list_[0],
			d_inextNodeNumListAssist_.list_[0],
			nextList_.size_);

	
}

// 把triNodeList与left/rightChildListFlag进行compact, 得到newTriNodeList:triIdx
void GPU_BuildKdTree::collectTrisToNewTriNodeList()
{
	//// 得到newTriNodeList中prims的数目
	size_t h_otemp;

	//cudaError_t e = cudaGetLastError();
	
	CUDA_SAFE_CALL(cudaMemcpy(&h_otemp, d_inextNodeNumListAssist_.list_[0] + nextList_.size_ - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&newTriNodeList_.size_, d_onextNodeNumListAssist_.list_[0] + nextList_.size_ - 1, sizeof(size_t), cudaMemcpyDeviceToHost));

	newTriNodeList_.size_ += h_otemp;

	


	//// 对左右孩子节点进行收集,compact
	// 对左孩子进行compact
	kdCompact(	CUDPP_UINT, 
		newTriNodeList_.list_[0],
		triNodeList_.list_[0],
		d_ileftChildListFlagsAssist_.list_[0],
		triNodeList_.size_);

	//e = cudaGetLastError();

	// 右孩子进行compact
	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));
	
	//e = cudaGetLastError();
	kdCompact(	CUDPP_UINT, 
		newTriNodeList_.list_[0] + (*h_numValid_),
		triNodeList_.list_[0],
		d_irightChildListFlagsAssist_.list_[0],
		triNodeList_.size_);
}



// 设置newTriNodeListAssist_:flag, 并建立nextList与newTriNodeListAssit_之间的联系, 并对segment进行初始化
void GPU_BuildKdTree::finishNextList()
{
	//// run kernal
	size_t blockSize = FINISH_NEXT_LIST_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * FINISH_NEXT_LIST_THREAD_HANDLE_NUM;

	// ***得到tri-node的大小，可估计
	size_t gridSize = (nextList_.size_ & (everyBlockNum - 1)) ? nextList_.size_ / everyBlockNum + 1 : nextList_.size_ / everyBlockNum;
	
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// set flags
	CUDA_SAFE_CALL(cudaMemset(d_ilargeNodeFlagsListAssist_.list_[0], 0, sizeof(size_t) * nextList_.size_));
	CUDA_SAFE_CALL(cudaMemset(d_ilargeNodeTriNodeFlagsListAssist_.list_[0], 0, sizeof(size_t) * newTriNodeList_.size_));
	CUDA_SAFE_CALL(cudaMemset(d_ismallNodeFlagsListAssist_.list_[0], 0, sizeof(size_t) * nextList_.size_));
	CUDA_SAFE_CALL(cudaMemset(d_ismallNodeTriNodeFlagsListAssist_.list_[0], 0, sizeof(size_t) * newTriNodeList_.size_));

	CUDA_SAFE_CALL(cudaMemset(newTriNodeList_.list_[1], 0, sizeof(size_t) * newTriNodeList_.size_));


	// run kernal
	if (gridSize > 0)
	{
	    finishNextListKernal<<<gDim, bDim, 0>>>(	nextList_.list_[0],
													//~kdNode_.list_[0],
													kdNodeBase_.list_[0],
													kdNodeBB_.list_[0],
													kdNodeExtra_.list_[0],
													//~kdNode_.size_,
													kdNodeBase_.size_,
													nextList_.size_,
													d_inextNodeNumListAssist_.list_[0],
													d_onextNodeNumListAssist_.list_[0],
													newTriNodeList_.list_[1],
													d_ilargeNodeFlagsListAssist_.list_[0],
													d_ilargeNodeTriNodeFlagsListAssist_.list_[0],
													d_ismallNodeFlagsListAssist_.list_[0],
													d_ismallNodeTriNodeFlagsListAssist_.list_[0]);
	}
	
}


// 对nextNodeNumList进行左右孩子节点的compact
void GPU_BuildKdTree::filter_extractNumList()
{
	//// large node compact
	kdCompact(	CUDPP_UINT, 
		d_ismallNodeNumListAssist_.list_[0],
		d_inextNodeNumListAssist_.list_[0],
		d_ismallNodeFlagsListAssist_.list_[0],
		nextList_.size_);

	// 得到smallNodeNextList大小
	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));
	smallNodeNextList_.size_ = *h_numValid_;

	if (smallNodeNextList_.size_ == 0)
	{
		isNeedFilter_ = false;
		return;
	}

	//// large node compact
	kdCompact(	CUDPP_UINT, 
		d_ilargeNodeNumListAssist_.list_[0],
		d_inextNodeNumListAssist_.list_[0],
		d_ilargeNodeFlagsListAssist_.list_[0],
		nextList_.size_);

	// 得到largeNodeNextList大小
	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));
	largeNodeNextList_.size_ = *h_numValid_;

	

}

// 对largeNodeNumList和smallNodeNumList进行exclusive scan，得到各自在triNode的起始地址
void GPU_BuildKdTree::filter_calTriNodeAddr()
{
	//// large node handle
	if (largeNodeNextList_.size_ > 0)
	{
	    kdScan(	CUDPP_ADD, 
				CUDPP_UINT,
				CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
				d_olargeNodeNumListAssist_.list_[0],
				d_ilargeNodeNumListAssist_.list_[0],
				largeNodeNextList_.size_);
	}
	

	//// small node handle
	if (smallNodeNextList_.size_ > 0)
	{
	    kdScan(	CUDPP_ADD, 
				CUDPP_UINT,
				CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
				d_osmallNodeNumListAssist_.list_[0],
				d_ismallNodeNumListAssist_.list_[0],
				smallNodeNextList_.size_);
	}
	
}

// 对nextList中的largeNode和smallNode分别进行compact
void GPU_BuildKdTree::filter_extractNextList()
{
	//// large node compact
	kdCompact(	CUDPP_UINT, 
		largeNodeNextList_.list_[0],
		nextList_.list_[0],
		d_ilargeNodeFlagsListAssist_.list_[0],
		nextList_.size_);

	//// small node compact
	kdCompact(	CUDPP_UINT, 
		smallNodeNextList_.list_[0] + smallNodeNextListSizeGlobal_,
		nextList_.list_[0],
		d_ismallNodeFlagsListAssist_.list_[0],
		nextList_.size_);
}

// 对large/small node triNodeFlagsList做segment scan把large/small的标志位1
void GPU_BuildKdTree::filter_distributeTriNodeFlags()
{
	//// distribute large node
	kdSegScan(	CUDPP_MAX,
		CUDPP_UINT,
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
		d_olargeNodeTriNodeFlagsListAssist_.list_[0],
		d_ilargeNodeTriNodeFlagsListAssist_.list_[0],
		newTriNodeList_.list_[1],
		newTriNodeList_.size_);

	//// distribute small node
	kdSegScan(	CUDPP_MAX,
		CUDPP_UINT,
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
		d_osmallNodeTriNodeFlagsListAssist_.list_[0], 
		d_ismallNodeTriNodeFlagsListAssist_.list_[0],
		newTriNodeList_.list_[1],
		newTriNodeList_.size_);
}

// 从newTriNodeList中分别提取出属于large/small节点的triIdx
void GPU_BuildKdTree::extractTriIdx()
{
	//// 提取large node的triIdx
	kdCompact(	CUDPP_UINT,
		largeNodeTriNodeList_.list_[0], 
		newTriNodeList_.list_[0],
		d_olargeNodeTriNodeFlagsListAssist_.list_[0],
		newTriNodeList_.size_);

	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));
	largeNodeTriNodeList_.size_ = *h_numValid_;

	//// 提取small node的triIdx
	kdCompact(	CUDPP_UINT,
		smallNodeTriNodeList_.list_[0] + smallNodeTriNodeListSizeGlobal_, 
		newTriNodeList_.list_[0],
		d_osmallNodeTriNodeFlagsListAssist_.list_[0],
		newTriNodeList_.size_);

	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));
	smallNodeTriNodeList_.size_ = *h_numValid_;

}

// 完成过滤，对kdNode指向的triNode重新定位
void GPU_BuildKdTree::filter_finish()
{
	// 对large/small的triNode:segflags进行预设置
	CUDA_SAFE_CALL(cudaMemset(largeNodeTriNodeList_.list_[1], 0, sizeof(size_t) * largeNodeTriNodeList_.size_));
	CUDA_SAFE_CALL(cudaMemset(smallNodeTriNodeList_.list_[1] + smallNodeTriNodeListSizeGlobal_, 0, sizeof(size_t) * smallNodeTriNodeList_.size_));

	// 对largeNodeNextList进行处理
	size_t blockSize = FILTER_FINISH_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * FILTER_FINISH_THREAD_HANDLE_NUM;
	size_t gridSize = (largeNodeNextList_.size_ & (everyBlockNum - 1)) ? largeNodeNextList_.size_ / everyBlockNum + 1 : largeNodeNextList_.size_ / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	if (gridSize > 0)
	{
	    finishNodeNextListKernal<<<gDim, bDim, 0>>>(	largeNodeNextList_.list_[0],
														largeNodeNextList_.size_,
														d_olargeNodeNumListAssist_.list_[0],
														largeNodeTriNodeList_.list_[1],
														//~kdNode_.list_[0],
														kdNodeBase_.list_[0],
														kdNodeBB_.list_[0],
														kdNodeExtra_.list_[0],
														0);
	}
	    

	// 对smallNodeNextList进行处理
	gridSize = (smallNodeNextList_.size_ & (everyBlockNum - 1)) ? smallNodeNextList_.size_ / everyBlockNum + 1 : smallNodeNextList_.size_ / everyBlockNum;
	gDim.x = gridSize;

	if (gridSize > 0)
	{
	    finishNodeNextListKernal<<<gDim, bDim, 0>>>(	smallNodeNextList_.list_[0] + smallNodeNextListSizeGlobal_,
														smallNodeNextList_.size_,
														d_osmallNodeNumListAssist_.list_[0],
														smallNodeTriNodeList_.list_[1] + smallNodeTriNodeListSizeGlobal_,
														//~kdNode_.list_[0],
														kdNodeBase_.list_[0],
														kdNodeBB_.list_[0],
														kdNodeExtra_.list_[0],
														smallNodeTriNodeListSizeGlobal_);
	}
	

	// 增加数目
	//printf("small node size: %d\n", smallNodeNextList_.size_);
	smallNodeNextListSizeGlobal_ += smallNodeNextList_.size_;
	smallNodeTriNodeListSizeGlobal_ += smallNodeTriNodeList_.size_;


}


void GPU_BuildKdTree::filter()
{
	//// 对nextNodeNumList进行左右孩子节点的compact
	filter_extractNumList();

	// 不需要进行filter
	if (!isNeedFilter_)
	{
		return;
	}

	//// 对largeNodeNumList和smallNodeNumList进行exclusive scan，得到各自在triNode的起始地址
	filter_calTriNodeAddr();

	//// 对nextList中的largeNode和smallNode分别进行compact
	filter_extractNextList();

	//// 对large/small node triNodeFlagsList做segment scan把large/small的标志位1
	filter_distributeTriNodeFlags();

	//// 从newTriNodeList中分别提取出属于large/small节点的triIdx
	extractTriIdx();

	//// 完成过滤，对kdNode指向的triNode重新定位
	filter_finish();
}
// 处理Large Node
void GPU_BuildKdTree::processLargeNode()
{
    //cudaError_t e;
	//isNeedTight_ = ((largeProcTime_ & (TIGHT_CYCLE - 1)) == 0 || largeProcTime_ == 0) ? 1 : 0;

	
	/*if (isNeedTight_)
	{*/
		//TIMER_START(t1);
		//// 收集triNodeList中所有node的BB，存入triNodePrimsBBList中
		collectTriNodePrimsBB();
		//TIMER_FINISH(t1, "???1???");

		//TIMER_START(t2);
		/*CuTimer t2;
		t2.startTimer();*/
		//// 计算activeList中每个node的包围盒
		calActiveNodeBoundingBox();
		//t2.finishTimer("???2???");
		//TIMER_FINISH(t2, "???2???");
	//}
	

	//TIMER_START(t3);
	//// 设置activeNodeList中所有node的boundingbox, splittype, splitValue, 并且设置其左右孩子节点的father, flag
	set_ActiveNode_Child_PreDistribute();
	// TIMER_FINISH(t3, "???3???");

	// TIMER_START(t4);
	//// distribute split,用于把prims分配到左右孩子节点中去
	distributeSplit();
	// TIMER_FINISH(t4, "???4???");

	// TIMER_START(t5);
	//// 以split设置tri-node中所有tri归属到左孩子还是右孩子
	setFlagWithSplit();
	// TIMER_FINISH(t5, "???5???");

	// TIMER_START(t6);
	//// 计算将来nextList中每个node中含有prim的数目
	calNextListNodePrimNum();
	// TIMER_FINISH(t6, "???6???");

	// TIMER_START(t7);
	//// 把left/rightChildListFlags中的代表next list node中prims数目的值提出出来放入nextNodeNumList
	extractPrimNumFromChildFlags();
	// TIMER_FINISH(t7, "???7???");

	// TIMER_START(t8);
	//// 对nextNodeNumList进行exclusive scan，得到nextNodeListNew在newTriNodeList中的起始地址
	/*printf("error code: %d", cudaGetLastError());
	
	printf("d_onextNodeNumListAssist_: %d\n", (void *)d_onextNodeNumListAssist_.list_[0]);
	printf("d_inextNodeNumListAssist_: %d\n", (void *)d_inextNodeNumListAssist_.list_[0]);
	printf("nextlist size: %d\n", nextList_.size_);*/
	calNewTriNodeListAddr();

	//printf("error code: %d", cudaGetLastError());
	// TIMER_FINISH(t8, "???8???");

	// TIMER_START(t9);
	//// 把triNodeList与left/rightChildListFlag进行compact, 得到newTriNodeList:triIdx
	collectTrisToNewTriNodeList();
	// TIMER_FINISH(t9, "???9???");

	// TIMER_START(t10);
	//// 设置newTriNodeListAssist_:flag, 并建立nextList与newTriNodeListAssit_之间的联系, 并对segment进行初始化
	finishNextList();
	// TIMER_FINISH(t10, "???10???");

	// TIMER_START(t11);
	//// large/small node filter
	/*CuTimer t11;
	t11.startTimer();*/
	filter();
	//t11.finishTimer();
	// TIMER_FINISH(t11, "???11???");

}

// 对small node进行预处理
void GPU_BuildKdTree::preProcessSmallNode()
{
	// 数据初始化,把从处理Large Node中得到的smallNodeNextList_赋给smallNodeList_
	smallNodeList_ = smallNodeNextList_;
	smallNodeList_.size_ = smallNodeNextListSizeGlobal_;
	smallNodeTriNodeList_.size_ = smallNodeTriNodeListSizeGlobal_;
	smallNodeRoot_ = smallNodeList_;


	// smallnode原始的数目(在处理完Large Node之后，Small Node之前)
	smallNodeOriginSize_ = smallNodeList_.size_;
	smallNodeTriNodeOriginSize_ = smallNodeTriNodeList_.size_;

	// 每个tri在任意一个轴向上都有两个包围盒的边界
	smallNodeBoundryCapacity_ = 2 * smallNodeTriNodeList_.size_;

	// smallNodeNextList_最多为当前smallNodeList_大小的两倍(一分为二)
	smallNodeNextList_.initList(smallNodeList_.size_ * 2);

    // *********初始化**************
	smallNodeRootMaskHi_.initList(smallNodeList_.size_);
	smallNodeRootMaskLo_.initList(smallNodeList_.size_);

	smallNodeMaskHi_.initList(smallNodeList_.size_);
	smallNodeMaskLo_.initList(smallNodeList_.size_);

	smallNodeNextListMaskHi_.initList(smallNodeList_.size_ * 2);
	smallNodeNextListMaskLo_.initList(smallNodeList_.size_ * 2);

	// smallNodeRootList_与smallNodeRList的区别：
	// smallNodeRootList_：标识在triNode列表中的tris到底是属于哪一个small Node
	// smallNodeRList_:    标识当前smallNodeList_中每一个small Node对应的原始small Node的索引号
	d_ismallNodeRootList_.initList(smallNodeBoundryCapacity_);
	d_osmallNodeRootList_.initList(smallNodeBoundryCapacity_);

	smallNodeRList_.initList(smallNodeList_.size_);
	smallNodeNextListRList_.initList(2 * smallNodeList_.size_);

	smallNodeBoundrySize_ = 2 * smallNodeTriNodeList_.size_;
	
	d_ismallNodeMaskLeftHi_.initList(smallNodeBoundrySize_);
	d_osmallNodeMaskLeftHi_.initList(smallNodeBoundrySize_);

	d_ismallNodeMaskLeftLo_.initList(smallNodeBoundrySize_);
	d_osmallNodeMaskLeftLo_.initList(smallNodeBoundrySize_);

	d_ismallNodeMaskRightHi_.initList(smallNodeBoundrySize_);
	d_osmallNodeMaskRightHi_.initList(smallNodeBoundrySize_);

	d_ismallNodeMaskRightLo_.initList(smallNodeBoundrySize_);
	d_osmallNodeMaskRightLo_.initList(smallNodeBoundrySize_);
	
	d_ismallNodeEveryLeafSize_.initList(smallNodeList_.size_);
    d_osmallNodeEveryLeafSize_.initList(smallNodeList_.size_);

	d_ismallNodeSegStartAddr_.initList(smallNodeBoundrySize_);
	d_osmallNodeSegStartAddr_.initList(smallNodeBoundrySize_);
    

	//// 对small node的所有node计算原始的mask
	smallNode_createOriginMask();

	//// 对smallNodeBoundry进行初始化
	smallNode_initBound();

	//// 对smallNodeBoundryValue在各个轴上进行sorting
	smallNode_boundrySorting();

	//// 对boundry的候选split的mask进行初始化，并进行seg scan(or and)
	smallNode_setBoundryMask();

	//// 释放资源

	//// 为正式处理small node进行初始化



	CUDA_SAFE_CALL(cudaMemcpy(smallNodeMaskHi_.list_[0], smallNodeRootMaskHi_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(smallNodeMaskLo_.list_[0], smallNodeRootMaskLo_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToDevice));
	

}

// 对small node的所有node计算原始的mask
void GPU_BuildKdTree::smallNode_createOriginMask()
{
	

	// 为所有候选splits分配segment flag, 首先所有标记都置0,在边界的地方再置1
	smallNodeBoundryFlags_.initList( 2 * smallNodeTriNodeList_.size_);
	CUDA_SAFE_CALL(cudaMemset(smallNodeBoundryFlags_.list_[0], 0u, sizeof(size_t) * 2 * smallNodeTriNodeList_.size_));

	// init absolutely position
	CUDA_SAFE_CALL(cudaMemset(d_ismallNodeSegStartAddr_.list_[0], 0, sizeof(size_t) * 2 * smallNodeTriNodeList_.size_));
	
	// 所有的root索引都置为0,便于将来distribute(MAX)
	CUDA_SAFE_CALL(cudaMemset(d_ismallNodeRootList_.list_[0], 0u, sizeof(size_t) * 2 * smallNodeTriNodeList_.size_));

	size_t blockSize = SET_SMALL_NODE_MASK_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * SET_SMALL_NODE_MASK_THREAD_HANDLE_NUM;
	size_t size = smallNodeList_.size_;
	size_t gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);
	if (gridSize > 0)
	{
		// 1.设置原始small node的位掩码: smallNodeRootMaskHi_, smallNodeRootMaskLo_
		// 2.设置边界标志,在边界开始处置1： smallNodeBoundryFlags_
		// 3.在triNode每个segment开始处设置当前small node的索引: smallNodeRootList_
		// 4.为原始small node的每个node设置索引，从0开始
	    setSmallNodeRootMaskKernal<<<gDim, bDim, 0>>>(		smallNodeRootMaskHi_.list_[0], 
															smallNodeRootMaskLo_.list_[0],
															smallNodeList_.list_[0],
															smallNodeBoundryFlags_.list_[0],
															//d_iboundryAddValue_,
															d_ismallNodeSegStartAddr_.list_[0],
															d_ismallNodeEveryLeafSize_.list_[0],
															d_ismallNodeRootList_.list_[0],
															smallNodeRList_.list_[0],
															//~kdNode_.list_[0],
															kdNodeBase_.list_[0],
															kdNodeBB_.list_[0],
															kdNodeExtra_.list_[0],
															smallNodeList_.size_);
	}

	// 对d_ismallNodeEveryLeafSize_进行exclusize scan, 
	
    kdScan(	CUDPP_ADD, 
		CUDPP_UINT,
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
		d_osmallNodeEveryLeafSize_.list_[0],
		d_ismallNodeEveryLeafSize_.list_[0],
		smallNodeList_.size_);

	// 对smallNodeBoundryAPos_进行distribute
	kdSegScan(CUDPP_MAX,
			  CUDPP_UINT,
			  CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			  d_osmallNodeSegStartAddr_.list_[0], 
			  d_ismallNodeSegStartAddr_.list_[0],
			  smallNodeBoundryFlags_.list_[0],
			  2 * smallNodeTriNodeList_.size_
			  );
}

// 对smallNodeBoundry进行初始化
void GPU_BuildKdTree::smallNode_initBound()
{
    //cudaError_t e;

	//// 计算每个tri在各自的node中的相对位置
	size_t *d_irelativePos;
	size_t *d_orelativePos;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_irelativePos, sizeof(size_t) * smallNodeTriNodeList_.size_));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_orelativePos, sizeof(size_t) * smallNodeTriNodeList_.size_));


	size_t blockSize = SET_DEVICE_MEM_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * SET_DEVICE_MEM_THREAD_HANDLE_NUM;
	size_t size = smallNodeTriNodeList_.size_;
	size_t gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);
	if (gridSize > 0)
	{
	    deviceMemsetKernal<<<gDim, bDim, 0>>>(d_irelativePos, 1u, size);
	}


	// segmented scan
	kdSegScan(	CUDPP_ADD, 
		CUDPP_UINT,
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
		d_orelativePos,
		d_irelativePos,
		smallNodeTriNodeList_.list_[1],
		smallNodeTriNodeList_.size_);


	//// boundry的各种初始化 
	smallNodeBoundryValue_.initList(smallNodeBoundrySize_);
	smallNodeBoundryRPos_.initList(smallNodeBoundrySize_);
	smallNodeBoundryType_.initList(smallNodeBoundrySize_);
	smallNodeBoundryTriIdx_.initList(smallNodeBoundrySize_);
	smallNodeBoundryAPos_.initList(smallNodeBoundrySize_);
	

	// init
	blockSize = INIT_BOUNDRY_BLOCK_SIZE;
	everyBlockNum = blockSize * INIT_BOUNDRY_THREAD_HANDLE_NUM;
	size = smallNodeBoundrySize_;
	gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	bDim.x = blockSize;
	gDim.x = gridSize;

	if (gridSize > 0)
	{
		// 1.设置候选split的值：smallNodeBOundryValue_
		// 2.设置每个候选split(所对应的tri)在每个segment中的相对位置(未经排序的)
		// 3.设置每个候选split的类型，左边界还是右边界
		// 4.设置每个候选split所属的tri的索引
	    initBoundryKernal<<<gDim, bDim, 0>>>(	//allBoundingBoxList_.primsBoundingBox_,
												allBoundingBoxList_.list_[0],
												allBoundingBoxList_.list_[1],

												smallNodeTriNodeList_.list_[0],
												//smallNodeBoundryValue_,
												smallNodeBoundryValue_.list_[0],
												smallNodeBoundryValue_.list_[1],
												smallNodeBoundryValue_.list_[2],

												//smallNodeBoundryRPos_,
												smallNodeBoundryRPos_.list_[0],
												smallNodeBoundryRPos_.list_[1],
												smallNodeBoundryRPos_.list_[2],

												//smallNodeBoundryType_,
												smallNodeBoundryType_.list_[0],
												smallNodeBoundryType_.list_[1],
												smallNodeBoundryType_.list_[2],

												//smallNodeBoundryTriIdx_,
												smallNodeBoundryTriIdx_.list_[0],
												smallNodeBoundryTriIdx_.list_[1],
												smallNodeBoundryTriIdx_.list_[2],

												//smallNodeBoundryAPos_,
												smallNodeBoundryAPos_.list_[0],
												smallNodeBoundryAPos_.list_[1],
												smallNodeBoundryAPos_.list_[2],

												d_orelativePos,
												d_osmallNodeSegStartAddr_.list_[0],
												smallNodeBoundrySize_);	
	}

	// error check
	//cudaError_t e1 = cudaGetLastError();
	
	//// set pos list
	kdSegScan(	CUDPP_MAX,
		CUDPP_UINT,
		CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
		d_osmallNodeRootList_.list_[0],
		d_ismallNodeRootList_.list_[0],
		smallNodeBoundryFlags_.list_[0],
		smallNodeBoundrySize_);

	// forget release
	cudaFree(d_irelativePos);
	cudaFree(d_orelativePos);
}

// 对smallNodeBoundryValue在各个轴上进行sorting
void GPU_BuildKdTree::smallNode_boundrySorting()
{
	TIMER_START(bst);
	//CuTimer bstimer;
	//bstimer.startTimer();
	//// 分别对smallNodeBoundryAPos进行排序, x, y, z
	size_t blockSize = 16;
	dim3 bDim(blockSize, 1, 1);
	size_t gridSize = (smallNodeList_.size_ & 15) ? smallNodeList_.size_ / 16 + 1 : smallNodeList_.size_ / 16;
	dim3 gDim(gridSize, 1, 1);
	dim3 bd(CTA_SIZE_RADIX, 1, 1);
	dim3 gd((smallNodeList_.size_  + NODE_PER_BLOCK_RADIX - 1)  / NODE_PER_BLOCK_RADIX, 1, 1);

	for (int i = 0; i < 3; i++)
	{
		// 可以考虑用多次的kernal调用
	    /*segParallelQuickSort<<<gDim, bDim, 128 * 16 * 7>>>(   smallNodeBoundryValue_.list_[i],  
															  smallNodeBoundryAPos_.list_[i], 
															  d_ismallNodeEveryLeafSize_.list_[0], 
															  d_osmallNodeEveryLeafSize_.list_[0], 
															  smallNodeList_.size_);*/
		segmentRadixSort<<<gd, bd, 512 * 4>>>((uint *)smallNodeBoundryValue_.list_[i],  
				  (uint *) smallNodeBoundryAPos_.list_[i], 
				  d_ismallNodeEveryLeafSize_.list_[0], 
				  d_osmallNodeEveryLeafSize_.list_[0], 
				  smallNodeList_.size_);
	}
	//bstimer.finishTimer("boundry sort");
	TIMER_FINISH(bst, "----------boundry sort ");

	//printDeviceMem(_FLOAT_, smallNodeBoundryValue_.list_[0], smallNodeBoundrySize_, "BoundryValue-------------: ");
	// 验证快速排序的正确性
	/*float *h_smallNodeBoundryValue = (float *)malloc(sizeof(float) * smallNodeBoundrySize_);
	cudaMemcpy(h_smallNodeBoundryValue, smallNodeBoundryValue_.list_[0], sizeof(float) * smallNodeBoundrySize_, cudaMemcpyDeviceToHost);
	size_t *h_osmallNodeEveryLeafSize = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	cudaMemcpy(h_osmallNodeEveryLeafSize, d_osmallNodeEveryLeafSize_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	size_t *h_ismallNodeEveryLeafSize = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	cudaMemcpy(h_ismallNodeEveryLeafSize, d_ismallNodeEveryLeafSize_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	printf("pos&num for sort: \n");
	for (int i = 0; i < smallNodeList_.size_; i++)
	{
	    printf("(%d, %d) ", h_ismallNodeEveryLeafSize[i], h_osmallNodeEveryLeafSize[i]);
	}
	printf("\n");
	
	for (int i = 0; i < 6; i++)
	{
		int from = h_osmallNodeEveryLeafSize[i] + 1;
		int to = h_osmallNodeEveryLeafSize[i] + h_ismallNodeEveryLeafSize[i];
		printf("seg: %d, from pos: %d-->%d\n", i, from - 1, to - 1);
		printf("%f, ", h_smallNodeBoundryValue[from-1]);
	    for (int j = from; j < to; j++)
		{
			printf("%f, ", h_smallNodeBoundryValue[j]);
			if (h_smallNodeBoundryValue[j] < h_smallNodeBoundryValue[j - 1])
			{
				printf("\n");
				printf("seg: %d, segNum: %d, segpos: %d, segValue: (%f, %f)sort error!\n", i, h_ismallNodeEveryLeafSize[i], j, h_smallNodeBoundryValue[j - 1], h_smallNodeBoundryValue[j]);
			}
		}
		printf("************\n");
	}*/
	/*size_t *h_ismallNodeEveryLeafSize = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	cudaMemcpy(h_ismallNodeEveryLeafSize, d_ismallNodeEveryLeafSize_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	printf("nums: \n");
	for (int i = 0; i < smallNodeList_.size_; i++)
	{
	    printf("%d ", h_ismallNodeEveryLeafSize[i]);
	}
	printf("\n");*/


	//// 设置左右mask
	blockSize = SET_LEFT_RIGHT_MASK_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * SET_LEFT_RIGHT_MASK_THREAD_HANDLE_NUM;
	size_t size = smallNodeBoundrySize_;
	gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	bDim.x = blockSize;
	gDim.x = gridSize;
	if (gridSize > 0)
	{
	    setLeftRightMaskKernal<<<gDim, bDim, 0>>>(	//d_ismallNodeMaskLeftHi_,	// split左边prims数目, 高位, 输出
													d_ismallNodeMaskLeftHi_.list_[0],
													d_ismallNodeMaskLeftHi_.list_[1],
													d_ismallNodeMaskLeftHi_.list_[2],

													//d_ismallNodeMaskLeftLo_,	// split左边prims数目, 低位, 输出 
													d_ismallNodeMaskLeftLo_.list_[0],
													d_ismallNodeMaskLeftLo_.list_[1],
													d_ismallNodeMaskLeftLo_.list_[2],

													//d_ismallNodeMaskRightHi_,	// split右边prims数目, 高位, 输出
													d_ismallNodeMaskRightHi_.list_[0],
													d_ismallNodeMaskRightHi_.list_[1],
													d_ismallNodeMaskRightHi_.list_[2],

													//d_ismallNodeMaskRightLo_,	// split右边prims数目, 低位, 输出
													d_ismallNodeMaskRightLo_.list_[0],
													d_ismallNodeMaskRightLo_.list_[1],
													d_ismallNodeMaskRightLo_.list_[2],

													//smallNodeBoundryRPos_,		// 相对位置
													smallNodeBoundryRPos_.list_[0],
													smallNodeBoundryRPos_.list_[1],
													smallNodeBoundryRPos_.list_[2],
													
													//smallNodeBoundryAPos_,		// 绝对位置
													smallNodeBoundryAPos_.list_[0],
													smallNodeBoundryAPos_.list_[1],
													smallNodeBoundryAPos_.list_[2],

													//smallNodeBoundryType_,		// 类型
													smallNodeBoundryType_.list_[0],
													smallNodeBoundryType_.list_[1],
													smallNodeBoundryType_.list_[2],

													d_osmallNodeRootList_.list_[0],
													smallNodeRootMaskHi_.list_[0],
													smallNodeRootMaskLo_.list_[0],
													smallNodeBoundrySize_
													
													// 
													/*smallNodeList_.list_[0],
													kdNode_.list_[0]*/
													);	
	}
	
}



// 对boundry的候选split的mask进行初始化，并进行seg scan(or and)
void GPU_BuildKdTree::smallNode_setBoundryMask()
{
	//// 进行seg scan, 用以计算每个split的左右prims的数目
	for (int i = 0; i < 3; i++)
	{
		// 对maskLeft进行OR操作
		kdSegScan(CUDPP_BIT_OR,
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			d_osmallNodeMaskLeftHi_.list_[i],
			d_ismallNodeMaskLeftHi_.list_[i],
			smallNodeBoundryFlags_.list_[0],
			smallNodeBoundrySize_);
		kdSegScan(CUDPP_BIT_OR,
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			d_osmallNodeMaskLeftLo_.list_[i],
			d_ismallNodeMaskLeftLo_.list_[i],
			smallNodeBoundryFlags_.list_[0],
			smallNodeBoundrySize_);

		// 对maskRight进行AND操作
		kdSegScan(CUDPP_BIT_AND,
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			d_osmallNodeMaskRightHi_.list_[i],
			d_ismallNodeMaskRightHi_.list_[i],
			smallNodeBoundryFlags_.list_[0],
			smallNodeBoundrySize_);
		kdSegScan(CUDPP_BIT_AND,
			CUDPP_UINT,
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			d_osmallNodeMaskRightLo_.list_[i],
			d_ismallNodeMaskRightLo_.list_[i],
			smallNodeBoundryFlags_.list_[0],
			smallNodeBoundrySize_);
	}
}

// 初始化处理small node的一些数据, 在正式处理small node之前
void GPU_BuildKdTree::smallNode_initProcessSmallNode()
{
	// alloc, 叶子节点的数目，不会超过prims的数目
	d_ismallNodeLeafFlags_.initList(2 * smallNodeOriginSize_);
	d_osmallNodeLeafFlags_.initList(2 * smallNodeOriginSize_);
	d_ismallNodeNoLeafFlags_.initList(2 * smallNodeOriginSize_);
    d_osmallNodeNoLeafFlags_.initList(2 * smallNodeOriginSize_);
	d_ismallNodeLeafSize_.initList(2 * smallNodeOriginSize_);
	d_osmallNodeLeafSize_.initList(2 * smallNodeOriginSize_);

	leafPrims_.initList(10 * nPrims_);

}

// 对leaf node 预处理
void GPU_BuildKdTree::smallNode_leafNodeFilter()
{

	// 对非叶子节点标记进行scan操作, 得到非叶子节点左右孩子的起始地址：d_osmallNodeNoLeafFlags_.list_[0]
	// 同样也可以得到非叶子节点的数目
    kdScan(	CUDPP_ADD,
			CUDPP_UINT, 
			CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
			d_osmallNodeNoLeafFlags_.list_[0],
			d_ismallNodeNoLeafFlags_.list_[0],
			smallNodeList_.size_);

	// 得到smallNodeNextList的大小
	CUDA_SAFE_CALL(cudaMemcpy(&smallNodeNextList_.size_, d_osmallNodeNoLeafFlags_.list_[0] + smallNodeList_.size_ - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    smallNodeNextList_.size_ = smallNodeNextList_.size_ * 2;

	// 对叶子节点进行compact操作, 得到每个叶子节点的中含有prims的数目列表: d_osmallNodeLeafSize_.list_[0]
	kdCompact(CUDPP_UINT,
		d_osmallNodeLeafSize_.list_[0],
		d_ismallNodeLeafSize_.list_[0],
		d_ismallNodeLeafFlags_.list_[0],
		smallNodeList_.size_);

	// 得到叶子节点的数目
	CUDA_SAFE_CALL(cudaMemcpy(h_numValid_, d_numValid_, sizeof(size_t), cudaMemcpyDeviceToHost));

	// 对叶子节点的所含prim的数目进行exclusive scan, 得到每个叶子节点的prims的存放地址
	if (*h_numValid_ > 0)
	{
		// d_ismallNodeLeafSize_list_[0]: 叶子节点中tris存放的地址
		kdScan(	CUDPP_ADD,
				CUDPP_UINT,
				CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE,
				d_ismallNodeLeafSize_.list_[0],
				d_osmallNodeLeafSize_.list_[0],
				*h_numValid_
				); 

	   // 对叶子节点的标志进行scan, 得到叶子节点中prims的存放地址和数目
		kdScan(	CUDPP_ADD,
				CUDPP_UINT,
				CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE,
				d_osmallNodeLeafFlags_.list_[0],
				d_ismallNodeLeafFlags_.list_[0],
				smallNodeList_.size_);
	}
	

	

}

// 进行split
void GPU_BuildKdTree::smallNode_split()
{
	TIMER_START(proSNKernal);
	//CuTimer proSNKernal;
	//proSNKernal.startTimer();
	// 为每个非叶子节点找到最佳的split
	size_t blockSize = PROCESS_SMALL_NODE_BLOCK_SIZE;
	size_t everyBlockNum = blockSize * PROCESS_SMALL_NODE_THREAD_HANDLE_NUM;
	/*size_t threadEachNode = 4;
	size_t threadShift = 2;*/
	size_t size = smallNodeList_.size_ ;
	size_t gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// for test
	/*size_t* h_flags = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	size_t* d_flags;
	cudaMalloc((void **)&d_flags, sizeof(size_t) * smallNodeList_.size_);*/

	if (gridSize > 0)
	{
		// 使用SAH方法选择split plane,使用位掩码及Fast bit counting能进行快速的
		// 统计左右孩子中tris的数目,不需要进行排序(总的只要进行一次排序即可)
	    processSmallNodeKernal<<<gDim, bDim, 0>>>( smallNodeMaskHi_.list_[0],					// mask
													smallNodeMaskLo_.list_[0],
													smallNodeRList_.list_[0],						// root
													smallNodeRootMaskHi_.list_[0],				// root mask
													smallNodeRootMaskLo_.list_[0],
	
													d_osmallNodeMaskLeftHi_.list_[0],
													d_osmallNodeMaskLeftHi_.list_[1],
													d_osmallNodeMaskLeftHi_.list_[2],

													d_osmallNodeMaskLeftLo_.list_[0],
													d_osmallNodeMaskLeftLo_.list_[1],
													d_osmallNodeMaskLeftLo_.list_[2],

													d_osmallNodeMaskRightHi_.list_[0],
													d_osmallNodeMaskRightHi_.list_[1],
													d_osmallNodeMaskRightHi_.list_[2],

													d_osmallNodeMaskRightLo_.list_[0],
													d_osmallNodeMaskRightLo_.list_[1],
													d_osmallNodeMaskRightLo_.list_[2],

													smallNodeBoundryAPos_.list_[0],
													smallNodeBoundryAPos_.list_[1],
													smallNodeBoundryAPos_.list_[2],

													smallNodeBoundryRPos_.list_[0],
													smallNodeBoundryRPos_.list_[1],
													smallNodeBoundryRPos_.list_[2],

													smallNodeRoot_.list_[0],
													smallNodeList_.list_[0],				// small node list
													d_ismallNodeLeafFlags_.list_[0],					// leaf flags
													d_ismallNodeNoLeafFlags_.list_[0],				// nonleaf flags
													d_ismallNodeLeafSize_.list_[0],					// leaf prims size
													
													smallNodeBoundryValue_.list_[0],
													smallNodeBoundryValue_.list_[1],
													smallNodeBoundryValue_.list_[2],

													//~kdNode_.list_[0],
													kdNodeBase_.list_[0],
													kdNodeBB_.list_[0],
													kdNodeExtra_.list_[0],
													tCost_,
													iCost_,								// intersection cost
													smallProcTime_,
													//threadShift,
													smallNodeList_.size_
												    );
	}

	//proSNKernal.finishTimer("process small node kernal ");
	TIMER_FINISH(proSNKernal, "--------process small node kernal");

	// for test
	/*size_t* h_leaf = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	cudaMemcpy(h_leaf, d_ismallNodeLeafFlags_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	size_t* h_noleaf = (size_t *)malloc(sizeof(size_t) * smallNodeList_.size_);
	cudaMemcpy(h_noleaf, d_ismallNodeNoLeafFlags_.list_[0], sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_flags, d_flags, sizeof(size_t) * smallNodeList_.size_, cudaMemcpyDeviceToHost);
	printf("leaf flag:\n");
	for (int i = 0; i < smallNodeList_.size_; i++)
	{
	    printf("%d ", h_leaf[i]);
	}
	printf("\nno leaf flag: \n");
	for (int i = 0; i < smallNodeList_.size_; i++)
	{
		printf("%d ", h_noleaf[i]);
	}
	printf("\nflags:\n");
	for (int i = 0; i < smallNodeList_.size_; i++)
	{
		printf("%d ", h_flags[i]);
	}*/



	// 对leaf node 预处理
	smallNode_leafNodeFilter();

	// filter
	blockSize = LEAF_FILTER_BLOCK_SIZE;
	everyBlockNum = blockSize * LEAF_FILTER_THREAD_HANDLE_NUM;
	size = smallNodeList_.size_;
	gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	bDim.x = blockSize;
	gDim.x = gridSize;

    

	if (gridSize > 0)
	{
	    leafNodeFilterKernal<<<gDim, bDim, 0>>>(    d_ismallNodeNoLeafFlags_.list_[0],	// non leaf
													d_osmallNodeNoLeafFlags_.list_[0],

													d_osmallNodeLeafFlags_.list_[0],		// leaf 
													d_osmallNodeLeafSize_.list_[0],
													d_ismallNodeLeafSize_.list_[0],

													leafPrims_.list_[0],
													leafPrims_.size_,

													smallNodeRoot_.list_[0],
													smallNodeList_.list_[0],			// node list
													smallNodeNextList_.list_[0],		// next node list

													smallNodeRList_.list_[0],				// root list		
													smallNodeNextListRList_.list_[0],		// next root list

													smallNodeMaskHi_.list_[0], 
													smallNodeMaskLo_.list_[0],
													smallNodeNextListMaskHi_.list_[0], 
													smallNodeNextListMaskLo_.list_[0],

													smallNodeTriNodeList_.list_[0],

													//~kdNode_.list_[0],
													//~kdNode_.size_,
													kdNodeBase_.list_[0],
													kdNodeBB_.list_[0],
													kdNodeExtra_.list_[0],
													kdNodeBase_.size_,

													//d_osmallNodeMaskLeftHi_,				// root split mask
													d_osmallNodeMaskLeftHi_.list_[0],
													d_osmallNodeMaskLeftHi_.list_[1],
													d_osmallNodeMaskLeftHi_.list_[2],

													//d_osmallNodeMaskLeftLo_,
													d_osmallNodeMaskLeftLo_.list_[0],
													d_osmallNodeMaskLeftLo_.list_[1],
													d_osmallNodeMaskLeftLo_.list_[2],

													//d_osmallNodeMaskRightHi_,
													d_osmallNodeMaskRightHi_.list_[0],
													d_osmallNodeMaskRightHi_.list_[1],
													d_osmallNodeMaskRightHi_.list_[2],

													//d_osmallNodeMaskRightLo_,
													d_osmallNodeMaskRightLo_.list_[0],
													d_osmallNodeMaskRightLo_.list_[1],
													d_osmallNodeMaskRightLo_.list_[2],

													smallNodeList_.size_
													);

	}
	
	// 重设leafSize, kdSize
	size_t temp;
	size_t leafNum = *h_numValid_;
	leafNum_ += leafNum;

	// ***在EmuDebug中，会出现问题，不知道什么原因
	if (leafNum > 0)
	{
	    cudaMemcpy(&temp, d_ismallNodeLeafSize_.list_[0] + leafNum - 1, sizeof(size_t), cudaMemcpyDeviceToHost);
		leafPrims_.size_ += temp;
		cudaMemcpy(&temp, d_osmallNodeLeafSize_.list_[0] + leafNum - 1, sizeof(size_t), cudaMemcpyDeviceToHost);
		leafPrims_.size_ += temp;
	}
	

	//~kdNode_.size_ += smallNodeNextList_.size_;	// 新产生的kd node的数目为非leaf节点的2倍
	kdNodeBase_.size_ += smallNodeNextList_.size_;
}

// 正式处理small node
void GPU_BuildKdTree::processSmallNode()
{
	// 进行split
	smallNode_split();
}


// 测试raytracing
void GPU_BuildKdTree::genRayTest(int num)
{
    // org
	// float4 org = {0.5f, 0.5f, 0.0f, 1.0f};
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_tHit, sizeof(float) * num));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dir, sizeof(float4) * num));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_hitTri, sizeof(int) * num));

	// seed
	CUDPPConfiguration configSeed;
	CUDPPHandle seedHandle;
	
	configSeed.algorithm = CUDPP_RAND_MD5; 
	configSeed.datatype = CUDPP_UINT;
	configSeed.options = 0;
	CUDPPResult seedResult = cudppPlan(&seedHandle, configSeed, 2 * num, 1, 0);

	if (CUDPP_SUCCESS != seedResult)
	{
		printf("seed create plan failed!\n");
		exit(-1);
	}
    
	if (CUDPP_SUCCESS !=cudppRandSeed(seedHandle, (unsigned int)time(NULL)))
	{
		printf("seed failed\n");
		exit(-1);
	}

	// rand
    CUDPPConfiguration configRand;
	CUDPPHandle randHandle;
	
	configRand.algorithm = CUDPP_RAND_MD5; 
	configRand.datatype = CUDPP_UINT;
	configRand.options = 0;
	CUDPPResult randResult = cudppPlan(&randHandle, configRand, 2 * num, 1, 0);

	if (CUDPP_SUCCESS != randResult)
	{
		printf("rand create plan failed!\n");
		exit(-1);
	}
    
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_rand, sizeof(size_t) * 2 * num));
	if (CUDPP_SUCCESS !=cudppRand(randHandle, d_rand, 2 * num))
	{
		printf("rand failed\n");
		exit(-1);
	}	
}

void GPU_BuildKdTree::sendRayTest(int num)
{
    // rand
	genRayTest(num);

	// ray trace
	size_t blockSize = 64;
	size_t everyBlockNum = blockSize * 1;
	size_t size = num;
	size_t gridSize = (size & (everyBlockNum - 1)) ? size / everyBlockNum + 1 : size / everyBlockNum;
	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);


	// raytracing kernal
	//rayTracingKernalTest<<<gDim, bDim, 0>>>(//~kdNode_.list_[0], 
	//										//~kdNode_.size_, 
	//										kdNodeBase_.list_[0],
	//											kdNodeBB_.list_[0],
	//											kdNodeExtra_.list_[0],
	//											kdNodeBase_.size_,
	//										leafPrims_.list_[0],
	//										prims_.d_vertex_,
	//										prims_.d_face_,

	//										d_rand,
	//										d_tHit,
	//										d_dir,
	//										d_hitTri,
	//					                    num
	//										);
}

// real ray tracing
void GPU_BuildKdTree::rayTrace()
{
    // 
	size_t pixelPerThread = 1;
	size_t blockSize = 64;
	size_t everyBlockNum = blockSize * pixelPerThread;
	size_t pixelNum = sceneParam_.view_.wid_ * sceneParam_.view_.hei_;
	size_t gridSize = (pixelNum & (everyBlockNum - 1)) ? pixelNum / everyBlockNum + 1 : pixelNum / everyBlockNum;

	dim3 bDim(blockSize, 1, 1);
	dim3 gDim(gridSize, 1, 1);

	// pixel buffer
	/*if (h_pixelBuf_ == NULL)
		h_pixelBuf_ = (float3 *)malloc(sizeof(float3) * pixelNum);
	if (d_pixelBuf_ == NULL)
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelBuf_, sizeof(float3) * pixelNum));*/
	/*CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelBuf_, sizeof(uchar4) * pixelNum));
	h_pixelBuf_ = ( uchar4 * )malloc( sizeof( uchar4 ) * pixelNum);*/
	
	

	// 场景参数
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)&d_sceneParam, &sceneParam_, sizeof(Scene_d)));
	/*SCENE_ scene1;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((char *)&d_scene, &scene1, sizeof(SCENE_)));*/

	/*const int range = 100;
	size_t *d_hitTris;
	float *d_hitDis;
	float *d_hitCos;*/

	//cudaError_t e = cudaGetLastError();
	//printf("error code: %d\n", e);

	// test tex
	/*float2* d_texUV;
	cudaMalloc((void**)&d_texUV, sizeof(float2) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_);
	float2* h_texUV = (float2 *)malloc(sizeof(float2) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_);
	uchar4* genPixel = (uchar4 *)malloc(sizeof(uchar4) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_);
	int* d_hitTri;
	cudaMalloc((void **)&d_hitTri, sizeof(int) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_);
	int* h_hitTri = (int *)malloc(sizeof(int) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_);*/


	// ray trace kernal
	rayTraceKernal<<<gDim, bDim, 0>>>(		//~kdNode_.list_[0], 
											//~kdNode_.size_, 
											kdNodeBase_.list_[0],
											kdNodeBB_.list_[0],
											kdNodeExtra_.list_[0],
											kdNodeBase_.size_,

											leafPrims_.list_[0],
											prims_.d_vertex_[keyframe_],
											prims_.d_face_[keyframe_],
											prims_.d_texture_[keyframe_],
											prims_.d_normal_[keyframe_],
											prims_.d_facenormal_[keyframe_],
											prims_.d_facetexture_[keyframe_],

											//d_pixelBuf_,
											devPixelBufPtr_,
											pixelPerThread,
											pixelNum
											/* d_texUV,
											 d_hitTri*/
											//d_hitTris, // 3.8
											//d_hitDis,
											//d_hitCos
											);
	//e = cudaGetLastError();
	//printf("error code: %d\n", e);

	// test texture
	// testTexture(tex_, ppmNum_);
	//cudaMemcpy(h_texUV, d_texUV, sizeof(float2) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_hitTri, d_hitTri, sizeof(int) * sceneParam_.view_.wid_ * sceneParam_.view_.hei_, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < sceneParam_.view_.wid_ * sceneParam_.view_.hei_; i++)
	//{
	//	if (h_hitTri[i] >= 0)
	//	{
	//		int mtlIndex = prims_.h_face_[h_hitTri[i]].w;
	//		if (mtlIndex < 0) 
	//		{
	//			genPixel[i].x = 0;
	//			genPixel[i].y = 0;
	//			genPixel[i].z = 0;
	//			genPixel[i].w = 255;
	//			continue;
	//		}
	//		//printf("mtlIndex: %d\n", mtlIndex);
	//		int width = tex_[mtlIndex].wid;
	//		int height = tex_[mtlIndex].hei;
	//		int ix = h_texUV[i].x * width;
	//		int iy = h_texUV[i].y * height;
	//		int index = iy * width + ix;
	//		genPixel[i] = tex_[mtlIndex].texPtr[index];
	//	}
	//	else
	//	{
	//		genPixel[i].x = 0;
	//		genPixel[i].y = 0;
	//		genPixel[i].z = 0;
	//		genPixel[i].w = 255;
	//	}
	//}

	//writepng("E:\\xialong\\dynamic\\testtexture\\render1.png", sceneParam_.view_.wid_, sceneParam_.view_.hei_, (unsigned char*)genPixel);



	// 
	//CUDA_SAFE_CALL(cudaMemcpy(h_pixelBuf_, d_pixelBuf_, sizeof(uchar4) * pixelNum, cudaMemcpyDeviceToHost));
	
	// test 
	/*uchar4 *h_pixelBuf;
	h_pixelBuf = (uchar4*)malloc(sizeof(uchar4) * 800 * 800);
	CUDA_SAFE_CALL( cudaMemcpy( h_pixelBuf, devPixelBufPtr_, 4 * 800 * 800, cudaMemcpyDeviceToHost) );
	int pixels = 0;
	for (int i = 0; i < pixelNum; i++)
	{
	    if (h_pixelBuf[i].x > 0 || h_pixelBuf[i].y || h_pixelBuf[i].z > 0)
		{
		    //printf("(%d, %d, %d)\n", h_pixelBuf[i].x, h_pixelBuf[i].y, h_pixelBuf[i].z);
		    pixels ++;
		}
	}
	printf("pixels: %d\n", pixels);*/
}

// scan
void GPU_BuildKdTree::kdScan(   CUDPPOperator  op,        //!< The numerical operator to be applied
							 CUDPPDatatype  datatype,  //!< The datatype of the input arrays
							 unsigned int   options,   //!< Options to configure the algorithm
							 void           *d_out,
							 void		   *d_in,
							 size_t         numElements
								)
{
	CUDPPConfiguration configScan;
	CUDPPHandle scanHandle;

	configScan.op = op;
	configScan.algorithm = CUDPP_SCAN;
	configScan.datatype = datatype;
	configScan.options = options;

	CUDPPResult scanResult = cudppPlan(&scanHandle, configScan, numElements, 1, 0);
	if (CUDPP_SUCCESS != scanResult)
	{
		printf("scan create plan failed!\n");
		exit(-1);
	}

	if (CUDPP_SUCCESS !=cudppScan(	scanHandle, 
		d_out, 
		d_in, 
		numElements))
	{
		printf("scan failed\n");
		exit(-1);
	}

	// destroy plan
	if (CUDPP_SUCCESS != cudppDestroyPlan(scanHandle))
	{
		printf("cudpp destroy plan failed!\n");
		exit(-1);
	}
}

// segmented scan
void GPU_BuildKdTree::kdSegScan(CUDPPOperator  op,        //!< The numerical operator to be applied
								CUDPPDatatype  datatype,  //!< The datatype of the input arrays
								unsigned int   options,   //!< Options to configure the algorithm
								void           *d_out,
								void		   *d_in,
								size_t         *d_flags,
								size_t         numElements)
{
	CUDPPConfiguration configSegScan;
	CUDPPHandle segScanHandle;

	configSegScan.op = op;
	configSegScan.algorithm = CUDPP_SEGMENTED_SCAN;
	configSegScan.datatype = datatype;
	configSegScan.options = options;

	//cudaError_t e = cudaGetLastError();
	//int *test;
	//CUDA_SAFE_CALL(cudaMalloc((void**)&test, sizeof(int) * 8));

	CUDPPResult segScanResult = cudppPlan(&segScanHandle, configSegScan, numElements, 1, 0);
	if (CUDPP_SUCCESS != segScanResult)
	{
		printf("segment scan create plan failed!\n");
		exit(-1);
	}

	if (CUDPP_SUCCESS !=cudppSegmentedScan(segScanHandle, d_out, d_in, 
		d_flags, numElements))
	{
		printf("segment scan failed\n");
		exit(-1);
	}

	// destroy plan
	if (CUDPP_SUCCESS != cudppDestroyPlan(segScanHandle))
	{
		printf("cudpp destroy plan failed!\n");
		exit(-1);
	}
}

// compact
void GPU_BuildKdTree::kdCompact(CUDPPDatatype  datatype,  //!< The datatype of the input arrays
								void           *d_out,
								void		   *d_in,
								size_t         *d_valid,
								size_t         numElements)
{
	CUDPPConfiguration compactConfig;
	CUDPPHandle compactHandle;

	compactConfig.algorithm = CUDPP_COMPACT;
	compactConfig.datatype = datatype;
	compactConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	CUDPPResult compactResult = cudppPlan(&compactHandle, compactConfig, numElements, 1, 0);
	if (CUDPP_SUCCESS != compactResult)
	{
		printf("compact create plan failed!\n");
		exit(-1);
	}

	if (CUDPP_SUCCESS !=cudppCompact(	compactHandle, 
		d_out, 
		d_numValid_, 
		d_in, 
		d_valid, 
		numElements))
	{
		printf("compact segment scan failed\n");
		exit(-1);
	}

	// destroy plan
	if (CUDPP_SUCCESS != cudppDestroyPlan(compactHandle))
	{
		printf("compact cudpp destroy plan failed!\n");
		exit(-1);
	}
}

// sort
void GPU_BuildKdTree::kdSort(CUDPPDatatype  datatype,
							 unsigned int   options,
							 void			*d_keys,
							 void           *d_values,
							 size_t         numElements)
{
	CUDPPConfiguration sortConfig;
	sortConfig.datatype = datatype;
	sortConfig.options = options;

	CUDPPHandle sortHandle;
	CUDPPResult sortPlan = cudppPlan(&sortHandle, sortConfig, numElements, 1, 0);
	if (sortPlan != CUDPP_SUCCESS)
	{
		printf("sort cudpp plan failed\n");
		exit(-1);
	}

	// run sort
	CUDPPResult sortResult = cudppSort(sortHandle, d_keys, d_values, 32, numElements);
	if (CUDPP_SUCCESS != sortResult)
	{
		printf("cudppSort fail!\n");
		exit(-1);
	}
}

void GPU_BuildKdTree::printDeviceMem(_TYPE_ type, void* d_mem, int size, char *title)
{
	char *h_cmem;
	unsigned char* h_ucmem;
	int *h_imem;
	unsigned int *h_uimem;
	float *h_fmem;
	
	printf("%s\n", title);
	switch(type)
	{
	case _CHAR_:
		h_cmem = (char *)d_mem;
		h_cmem = (char *)malloc(sizeof(char) * size);
		cudaMemcpy(h_cmem, d_mem, sizeof(char) * size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < size; i++)
		{
			printf("%d, ", h_cmem[i]);
		}
		free(h_cmem);
		break;
	case _UCHAR_:
		h_ucmem = (unsigned char *)d_mem;
		h_ucmem = (unsigned char *)malloc(sizeof(unsigned char) * size);
		cudaMemcpy(h_ucmem, d_mem, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < size; i++)
		{
			printf("%d, ", h_ucmem[i]);
		}
		free(h_ucmem);
		break;
	case _INT_:
		h_imem = (int *)d_mem;
		h_imem = (int *)malloc(sizeof(int) * size);
		cudaMemcpy(h_imem, d_mem, sizeof(int) * size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < size; i++)
		{
			printf("%d, ", h_imem[i]);
		}
		free(h_imem);
		break;
	case _UINT_:
		h_uimem = (unsigned int *)d_mem;
		h_uimem = (unsigned int *)malloc(sizeof(unsigned int) * size);
		cudaMemcpy(h_uimem, d_mem, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < size; i++)
		{
			printf("%d, ", h_uimem[i]);
		}
		free(h_uimem);
		break;
	case _FLOAT_:
		h_fmem = (float *)d_mem;
		h_fmem = (float *)malloc(sizeof(float) * size);
		cudaMemcpy(h_fmem, d_mem, sizeof(float) * size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < size; i++)
		{
			printf("%f, ", h_fmem[i]);
		}
		free(h_fmem);
		break;
	default:
		break;
	}

	printf("\n");
}


//#endif