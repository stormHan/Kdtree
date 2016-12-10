#ifndef _KDGPU_APP_H_
#define _KDGPU_APP_H_
#include "kdgpu_data.h"
//#include "RayTrace/Workspace.h"
#include "dynamic_scene.h"

#include <vector>

using namespace std;

struct Node
{
	int *list_;
	size_t	size_;
};

typedef enum {_CHAR_, _UCHAR_, _INT_, _UINT_, _FLOAT_}_TYPE_;

struct GPU_BuildKdTree
{
	// 
	char *resultPath_;
	// **************data member****************
    size_t allocVideoMem_;
	// cost
	float iCost_;
	float tCost_;
	//// prims
	Prims  prims_;		// 物体
	size_t nPrims_;		// 物体数目
	size_t nVertex_;

	size_t keyframe_;
	size_t keynum_;

	

	//// basic list
	//~List<KdNodeOrigin, 1> kdNode_;			//KdNodeOrigin	*kdNode_;
	List<KdNode_base, 1>kdNodeBase_;
	List<KdNode_bb, 1>kdNodeBB_;
	List<KdNode_extra, 1>kdNodeExtra_;


	vector <Node>	nodeList_;
	List<int, 1> activeList_;	//KdNodeOrigin	**activeList_;
	List<int, 1> nextList_;		//KdNodeOrigin	**nextList_;


	size_t largeProcTime_;
	size_t isNeedTight_;
	List<size_t, 2> triNodeList_;		//TriNodeList        triNodeList_;			// tri-node associative list
	List<float4, 2> allBoundingBoxList_;	//AllBoundingBoxList allBoundingBoxList_;		// 所有物体的包围盒

	List<size_t, 2> newTriNodeList_;	//TriNodeList		newTriNodeList_;				// 用来辅助记录nextList相对应的tri-node关联

	List<int, 1> largeNodeNextList_;//KdNodeOrigin	**largeNodeNextList_;		// large node的nextList
	//size_t			largeNodeNextListSize_;
	List<size_t, 2> largeNodeTriNodeList_;//TriNodeList		largeNodeTriNodeList_;		// large node相关的triNode
	//size_t			largeNodeTriNodeListSize_;

	List<int, 1> smallNodeNextList_;	//KdNodeOrigin	**smallNodeNextList_;		// small node的nextList
	//size_t			smallNodeNextListSize_;
	size_t			smallNodeNextListSizeGlobal_;
	List<size_t, 2> smallNodeTriNodeList_;//TriNodeList		smallNodeTriNodeList_;		// small node相关的triNode
	//size_t			smallNodeTriNodeListSize_;
	size_t			smallNodeTriNodeListSizeGlobal_;

	//// size
	//size_t kdSize_;				// 已经产生kdNode的数目
	//size_t activeListSize_;		// activeList的大小
	//size_t nextListSize_;		// nextList的大小
	//size_t smallListSize_;		// smallList的大小

	//size_t triNodeListSize_;			// triNodeList的大小
	//size_t triNodePrimsBBListSize_;		// triNodePrimsBBList的大小

	//size_t newTriNodeListSize_;			// newTriNodeListAssist的大小




	//// assist list
	List<float, 6> d_itriNodePrimsBBListAssist_;	//float   *d_itriNodePrimsBBListAssist_[6];
	List<float, 6> d_otriNodePrimsBBListAssist_;	//float	*d_otriNodePrimsBBListAssist_[6];	// triNodePrimsBBList执行segmented scan的输出

	List<float, 1> d_isplitListAssist_;		//float	*d_isplitListAssist_;				// split value的segment scan输入
	List<float, 1> d_osplitListAssist_;		//float	*d_osplitListAssist_;				// split value的segment scan输出
	List<size_t, 1> d_isplitTypeListAssist_;	//size_t	*d_isplitTypeListAssist_;			// split type list的segment scan输入
	List<size_t, 1> d_osplitTypeListAssist_;    //size_t	*d_osplitTypeListAssist_;			// split type list的segment scan输出

	List<size_t, 1> d_ileftChildListFlagsAssist_;	//size_t  *d_ileftChildListFlagsAssist_;		// 区分左孩子的标志，用以segment scan输入
	List<size_t, 1> d_oleftChildListFlagsAssist_;	//size_t  *d_oleftChildListFlagsAssist_;		// 区分左孩子的标志，用以segment scan输出
	List<size_t, 1> d_irightChildListFlagsAssist_;	//size_t  *d_irightChildListFlagsAssist_;		// 区分右孩子的标志，用以segment scan输入
	List<size_t, 1> d_orightChildListFlagsAssist_;	//size_t  *d_orightChildListFlagsAssist_;		// 区分右孩子的标识，用以segment scan输出

	List<size_t, 1> d_inextNodeNumListAssist_;		//size_t  *d_inextNodeNumListAssist_;			// next node中节点中含有prim的个数，用以segment scan输入
	List<size_t, 1> d_onextNodeNumListAssist_;		//size_t  *d_onextNodeNumListAssist_;			// next node中节点中含有prim的个数，用以segment scan输出

	List<size_t, 1> d_ilargeNodeFlagsListAssist_;			//size_t		*d_ilargeNodeFlagsListAssist_;			// large node在nextNodeList中的标志, 输入
	List<size_t, 1> d_olargeNodeFlagsListAssist_;			//size_t		*d_olargeNodeFlagsListAssist_;			// large node在nextNodeList中的标志, 输出
	List<size_t, 1> d_ilargeNodeNumListAssist_;			//size_t		*d_ilargeNodeNumListAssist_;			// large node在nextNodeNumListAssist中的标志，输入
	List<size_t, 1> d_olargeNodeNumListAssist_;			//size_t		*d_olargeNodeNumListAssist_;			// large node在nextNodeNumListAssist中的标志，输出
	List<size_t, 1> d_ilargeNodeTriNodeFlagsListAssist_;	//size_t		*d_ilargeNodeTriNodeFlagsListAssist_;	// large node在newTriNodeList中的标志, 输入
	List<size_t, 1> d_olargeNodeTriNodeFlagsListAssist_;	//size_t		*d_olargeNodeTriNodeFlagsListAssist_;	// large node在newTriNodeList中的标志, 输出

	List<size_t, 1> d_ismallNodeFlagsListAssist_;			//size_t		*d_ismallNodeFlagsListAssist_;			// small node在nextNodeList中的标志, 输入
	List<size_t, 1> d_osmallNodeFlagsListAssist_;			//size_t		*d_osmallNodeFlagsListAssist_;			// small node在nextNodeList中的标志, 输出
	List<size_t, 1> d_ismallNodeNumListAssist_;			//size_t		*d_ismallNodeNumListAssist_;			// small node在nextNodeNumListAssist中的标志，输入
	List<size_t, 1> d_osmallNodeNumListAssist_;			//size_t		*d_osmallNodeNumListAssist_;			// small node在nextNodeNumListAssist中的标志，输出
	List<size_t, 1> d_ismallNodeTriNodeFlagsListAssist_;	//size_t		*d_ismallNodeTriNodeFlagsListAssist_;	// small node在newTriNodeList中的标志, 输入
	List<size_t, 1> d_osmallNodeTriNodeFlagsListAssist_;	//size_t		*d_osmallNodeTriNodeFlagsListAssist_;	// small node在newTriNodeList中的标志, 输出



	//// temp
	size_t	*d_numValid_;		// 进行compact，1的个数, device mem
	size_t  *h_numValid_;		// 进行compact, 1的个数, host   mem


	/**************small node handle***************/
	List<int, 1> smallNodeList_; //KdNodeOrigin **smallNodeList_;			// small node
	//size_t       smallNodeListSize_;		// 大小
	List<size_t, 1> leafPrims_; //size_t       *leafPrims_;		// 所有叶子节点所包含的prims
	//size_t       leafSize_;
	size_t       leafNum_;
	size_t       smallProcTime_;

	List<int, 1> smallNodeRoot_;	//KdNodeOrigin **smallNodeRoot_;			// 原始的small root
	List<size_t, 1> d_ismallNodeRootList_;		//size_t *d_ismallNodeRootList_;			// split所属small node root列表, 输入
	List<size_t, 1> d_osmallNodeRootList_;		//size_t *d_osmallNodeRootList_;			// split所属small node root列表, 输出	

	List<size_t, 1> smallNodeRootMaskHi_;		//size_t *smallNodeRootMaskHi_;			// 原始的small root的mask的高端有效位
	List<size_t, 1> smallNodeRootMaskLo_;		//size_t *smallNodeRootMaskLo_;			// 原始的small root的mask的低端有效位

	size_t smallNodeBoundrySize_;			// 候选split的数目
	List<float, 3> smallNodeBoundryValue_;  //float  *smallNodeBoundryValue_[3];		// 边界值 x, y, z, 为了进行segsort, 人工进行了加值
	List<size_t, 3> smallNodeBoundryRPos_;  //size_t *smallNodeBoundryRPos_[3];		// boundry相对于segflag位置
	List<size_t, 1> smallNodeBoundryFlags_;    //size_t *smallNodeBoundryFlags_;			// boundry的flag, 开始出位1, 其它为0
	List<size_t, 3> smallNodeBoundryType_;  //size_t *smallNodeBoundryType_[3];		// boundry的类型, 开始的位置为1, 结束位0
	List<size_t, 3> smallNodeBoundryTriIdx_;//size_t *smallNodeBoundryTriIdx_[3];		// boundry所属的tri
	List<size_t, 3> smallNodeBoundryAPos_;  //size_t *smallNodeBoundryAPos_[3];		// boundry的绝对位置, 输入

	// 原始small node列表中每一个node中包含的tris的数目，scan之后可以用来进行地址运算
	List<size_t, 1> d_ismallNodeEveryLeafSize_; 
	List<size_t, 1> d_osmallNodeEveryLeafSize_; //size_t *d_osmallNodeEveryLeafSize_;
	
	// ??????????
	List<size_t, 1> d_ismallNodeSegStartAddr_;  //size_t *d_ismallNodeSegStartAddr_;
	List<size_t, 1> d_osmallNodeSegStartAddr_;  //size_t *d_osmallNodeSegStartAddr_;

	List<size_t, 3> d_ismallNodeMaskLeftHi_; //size_t *d_ismallNodeMaskLeftHi_[3];		// split左边prims数目, 高位, 输入
	List<size_t, 3> d_osmallNodeMaskLeftHi_; //size_t *d_osmallNodeMaskLeftHi_[3];		// split左边prims数目, 高位, 输出
	List<size_t, 3> d_ismallNodeMaskLeftLo_; //size_t *d_ismallNodeMaskLeftLo_[3];		// split左边prims数目, 低位, 输入
	List<size_t, 3> d_osmallNodeMaskLeftLo_; //size_t *d_osmallNodeMaskLeftLo_[3];		// split左边prims数目, 低位, 输出 

	List<size_t, 3> d_ismallNodeMaskRightHi_; //size_t *d_ismallNodeMaskRightHi_[3];	// split右边prims数目, 高位, 输入
	List<size_t, 3> d_osmallNodeMaskRightHi_; //size_t *d_osmallNodeMaskRightHi_[3];	// split右边prims数目, 高位, 输出
	List<size_t, 3> d_ismallNodeMaskRightLo_; //size_t *d_ismallNodeMaskRightLo_[3];	// split右边prims数目, 低位, 输入
	List<size_t, 3> d_osmallNodeMaskRightLo_; //size_t *d_osmallNodeMaskRightLo_[3];	// split右边prims数目, 低位, 输出 

	size_t smallNodeBoundryCapacity_;
	List<size_t, 1> smallNodeMaskHi_; //size_t *smallNodeMaskHi_;				// small node mask hi
	List<size_t, 1> smallNodeMaskLo_; //size_t *smallNodeMaskLo_;				// small node mask lo

	List<size_t, 1> smallNodeNextListMaskHi_; //size_t *smallNodeNextListMaskHi_;
	List<size_t, 1> smallNodeNextListMaskLo_; //size_t *smallNodeNextListMaskLo_;

	List<size_t, 1> smallNodeRList_;   //size_t *smallNodeRList_;

	List<size_t, 1> smallNodeNextListRList_; //size_t *smallNodeNextListRList_;

	List<size_t, 1> d_ismallNodeLeafFlags_; //size_t *d_ismallNodeLeafFlags_;			// 把叶子节点标记为1, 输入, 用来做d_ismallNodeLeafSize_ compact的isValid,
	List<size_t, 1> d_osmallNodeLeafFlags_; //size_t *d_osmallNodeLeafFlags_;			// 把叶子节点标记为1, 输出, 经过scan ADD之后, 得到d_osmallNodeLeafSize_中leaf的大小的位置

	List<size_t, 1> d_ismallNodeNoLeafFlags_; //size_t *d_ismallNodeNoLeafFlags_;		// 把非叶子节点标记为1, 输入
	List<size_t, 1> d_osmallNodeNoLeafFlags_; //size_t *d_osmallNodeNoLeafFlags_;		// 把非叶子节点标记为1, 输出

	List<size_t, 1> d_ismallNodeLeafSize_; //size_t *d_ismallNodeLeafSize_;			// 叶子节点中含有prim的数目, compact之前
	List<size_t, 1> d_osmallNodeLeafSize_; //size_t *d_osmallNodeLeafSize_;			// 叶子节点中含有prim的数目, compact之后, 用于得到叶子节点所包含的prims的起始地址


	size_t smallNodeOriginSize_;
	size_t smallNodeTriNodeOriginSize_;

	bool isNeedFilter_;

	/*********ray tracing**/
	

	// print device mem
	void printDeviceMem(_TYPE_ type, void* d_mem, int size, char *title);

	// print bb
	void saveBB(char* path);

	// **************fun member******************
#ifdef NEED_TEXTURE
	//// 纹理
	Tex* tex_;
	int ppmNum_;
	void refTexture(Tex *tex, int ppmNum);
	void GPU_BuildKdTree::bindTexture2D(uchar4* h_gtex, uint4* h_texPos);
	void GPU_BuildKdTree::testTexture(Tex* tex, int ppmNum);
#endif

	//// 初始化
	void GPU_BuildKdTree::globalInit(SceneInfoArr &scene, int keynum, 
									 uchar4* h_gtex, uint4* h_texPos);
	void GPU_BuildKdTree::initialization();

	//// scan
	void kdScan(	CUDPPOperator  op,        //!< The numerical operator to be applied
		CUDPPDatatype  datatype,  //!< The datatype of the input arrays
		unsigned int   options,   //!< Options to configure the algorithm
		void           *d_out,
		void		   *d_in,
		size_t         numElements);

	//// segmented scan
	void kdSegScan(	CUDPPOperator  op,        //!< The numerical operator to be applied
		CUDPPDatatype  datatype,  //!< The datatype of the input arrays
		unsigned int   options,   //!< Options to configure the algorithm
		void           *d_out,
		void		   *d_in,
		size_t         *d_flags,
		size_t         numElements);

	//// compact
	void kdCompact(	CUDPPDatatype  datatype,  //!< The datatype of the input arrays
		void           *d_out,
		void		   *d_in,
		size_t         *d_valid,
		size_t         numElements);

	// sort
	void kdSort( CUDPPDatatype  datatype,
		unsigned int   options,
		void			*d_keys,
		void           *d_values,
		size_t         numElements);



	//// kd tree 构建
	void buildTree();

	//// list 交换
	template <typename T, int dimension>
    void swapList(List<T, dimension> &list1, List<T, dimension> &list2);

	//// 把activeList添加到nodeList
	void largeNodeAppendNodeList();

	//// 把largeNodeList与active进行交换
	void largeNodeSwap();

	//// 对largeNode处置中的一些内存进行扩展(动态表)
	void largeNodeExtendMem();

	//// 创建根节点
	void createRoot();

	//// 计算prims的包围盒
	void calPrimsBoundingBox();

	//// 处理large node
	void processLargeNode();

	//// 收集triNodeList中所有node的BB,存入triNodePrimsBBList中去，
	void collectTriNodePrimsBB();

	//// 计算activeList中每个node的包围盒,以triNodePrimsBBList作为data,triNodeList:segFlags作为flags, 执行segmented scan
	void calActiveNodeBoundingBox();

	//// 设置activeNodeList中所有node的boundingbox, splittype, splitValue,为distribute做准备
	void set_ActiveNode_Child_PreDistribute();

	//// 把activeNodeList中所有node的splitvalue, splittype，分配到辅助list中
	void distributeSplit();

	//// 以split设置tri-node中所有tri归属到左孩子还是右孩子
	void setFlagWithSplit();

	//// 计算将来nextList中每个node中含有prim的数目
	void calNextListNodePrimNum();

	//// 把left/rightChildListFlags中的代表next list node中prims数目的值提出出来放入nextNodeNumList
	void extractPrimNumFromChildFlags();

	//// 对nextNodeNumList进行exclusive scan，得到nextNodeListNew在newTriNodeList中的起始地址
	void calNewTriNodeListAddr();

	//// 把triNodeList与left/rightChildListFlag进行compact, 得到newTriNodeList:triIdx
	void collectTrisToNewTriNodeList();

	//// 设置newTriNodeListAssist_:flag, 并建立nextList与newTriNodeListAssit_之间的联系, 并对segment进行初始化
	void finishNextList();

	//// 是否处理large node时开辟的空间
    void largeNodeRelease();

	/********** small node filter*************/
	//// 对nextNodeNumList进行左右孩子节点的compact
	void filter_extractNumList();

	//// 对largeNodeNumList和smallNodeNumList进行exclusive scan，得到各自在triNode的起始地址
	void filter_calTriNodeAddr();

	//// 对nextList中的largeNode和smallNode分别进行compact
	void filter_extractNextList();

	//// 对large/small node triNodeFlagsList做segment scan把large/small的标志位1
	void filter_distributeTriNodeFlags();

	//// 从newTriNodeList中分别提取出属于large/small节点的triIdx
	void extractTriIdx();

	//// 完成过滤，对kdNode指向的triNode重新定位
	void filter_finish();

	//// filter全过程
	void filter();

	/********** small node handle **************/
	//// 对small node进行预处理
	void preProcessSmallNode();

	//// 对small node的所有node计算原始的mask
	void smallNode_createOriginMask();

	//// 对smallNodeBoundry进行初始化
	void smallNode_initBound();

	//// 对smallNodeBoundryValue在各个轴上进行sorting
	void smallNode_boundrySorting();

	//// 对boundry的候选split的mask进行初始化，并进行seg scan(or and)
	void smallNode_setBoundryMask();

	//// 初始化处理small node的一些数据, 在正式处理small node之前
	void smallNode_initProcessSmallNode();

	//// 对leaf node 预处理
	void smallNode_leafNodeFilter();

	//// 进行split
	void smallNode_split();

	//// 正式进行small node进行处理
	void processSmallNode();

	//// 把smallNodeList与smallNodeNextList进行交换
	void smallNodeSwap();

	//// 对smallNode处置中的一些内存进行扩展(动态表)
	void smallNodeExtendMem();

	//// 把smallNodeList添加到nodeList中
	void smallNodeAppendNodeList();

	//// 释放处理small node的显存资源
	void smallNodeRelease();
	/******************save kd tree*************/
	void saveKdTree();
	void readKdTree();

	/******************ray tracing******************/
	size_t* d_rand;
	float* d_tHit;
	float4* d_dir;
	int* d_hitTri;


	void genRayTest(int num);
	void sendRayTest(int num);
	
	// 光线跟踪
	// 场景参数
	Scene_d sceneParam_;
	uchar4 *d_pixelBuf_;
	uchar4 *h_pixelBuf_;
	uchar4* devPixelBufPtr_;
	void rayTrace();
    



};


#endif