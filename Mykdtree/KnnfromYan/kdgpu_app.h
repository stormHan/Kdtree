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
	Prims  prims_;		// ����
	size_t nPrims_;		// ������Ŀ
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
	List<float4, 2> allBoundingBoxList_;	//AllBoundingBoxList allBoundingBoxList_;		// ��������İ�Χ��

	List<size_t, 2> newTriNodeList_;	//TriNodeList		newTriNodeList_;				// ����������¼nextList���Ӧ��tri-node����

	List<int, 1> largeNodeNextList_;//KdNodeOrigin	**largeNodeNextList_;		// large node��nextList
	//size_t			largeNodeNextListSize_;
	List<size_t, 2> largeNodeTriNodeList_;//TriNodeList		largeNodeTriNodeList_;		// large node��ص�triNode
	//size_t			largeNodeTriNodeListSize_;

	List<int, 1> smallNodeNextList_;	//KdNodeOrigin	**smallNodeNextList_;		// small node��nextList
	//size_t			smallNodeNextListSize_;
	size_t			smallNodeNextListSizeGlobal_;
	List<size_t, 2> smallNodeTriNodeList_;//TriNodeList		smallNodeTriNodeList_;		// small node��ص�triNode
	//size_t			smallNodeTriNodeListSize_;
	size_t			smallNodeTriNodeListSizeGlobal_;

	//// size
	//size_t kdSize_;				// �Ѿ�����kdNode����Ŀ
	//size_t activeListSize_;		// activeList�Ĵ�С
	//size_t nextListSize_;		// nextList�Ĵ�С
	//size_t smallListSize_;		// smallList�Ĵ�С

	//size_t triNodeListSize_;			// triNodeList�Ĵ�С
	//size_t triNodePrimsBBListSize_;		// triNodePrimsBBList�Ĵ�С

	//size_t newTriNodeListSize_;			// newTriNodeListAssist�Ĵ�С




	//// assist list
	List<float, 6> d_itriNodePrimsBBListAssist_;	//float   *d_itriNodePrimsBBListAssist_[6];
	List<float, 6> d_otriNodePrimsBBListAssist_;	//float	*d_otriNodePrimsBBListAssist_[6];	// triNodePrimsBBListִ��segmented scan�����

	List<float, 1> d_isplitListAssist_;		//float	*d_isplitListAssist_;				// split value��segment scan����
	List<float, 1> d_osplitListAssist_;		//float	*d_osplitListAssist_;				// split value��segment scan���
	List<size_t, 1> d_isplitTypeListAssist_;	//size_t	*d_isplitTypeListAssist_;			// split type list��segment scan����
	List<size_t, 1> d_osplitTypeListAssist_;    //size_t	*d_osplitTypeListAssist_;			// split type list��segment scan���

	List<size_t, 1> d_ileftChildListFlagsAssist_;	//size_t  *d_ileftChildListFlagsAssist_;		// �������ӵı�־������segment scan����
	List<size_t, 1> d_oleftChildListFlagsAssist_;	//size_t  *d_oleftChildListFlagsAssist_;		// �������ӵı�־������segment scan���
	List<size_t, 1> d_irightChildListFlagsAssist_;	//size_t  *d_irightChildListFlagsAssist_;		// �����Һ��ӵı�־������segment scan����
	List<size_t, 1> d_orightChildListFlagsAssist_;	//size_t  *d_orightChildListFlagsAssist_;		// �����Һ��ӵı�ʶ������segment scan���

	List<size_t, 1> d_inextNodeNumListAssist_;		//size_t  *d_inextNodeNumListAssist_;			// next node�нڵ��к���prim�ĸ���������segment scan����
	List<size_t, 1> d_onextNodeNumListAssist_;		//size_t  *d_onextNodeNumListAssist_;			// next node�нڵ��к���prim�ĸ���������segment scan���

	List<size_t, 1> d_ilargeNodeFlagsListAssist_;			//size_t		*d_ilargeNodeFlagsListAssist_;			// large node��nextNodeList�еı�־, ����
	List<size_t, 1> d_olargeNodeFlagsListAssist_;			//size_t		*d_olargeNodeFlagsListAssist_;			// large node��nextNodeList�еı�־, ���
	List<size_t, 1> d_ilargeNodeNumListAssist_;			//size_t		*d_ilargeNodeNumListAssist_;			// large node��nextNodeNumListAssist�еı�־������
	List<size_t, 1> d_olargeNodeNumListAssist_;			//size_t		*d_olargeNodeNumListAssist_;			// large node��nextNodeNumListAssist�еı�־�����
	List<size_t, 1> d_ilargeNodeTriNodeFlagsListAssist_;	//size_t		*d_ilargeNodeTriNodeFlagsListAssist_;	// large node��newTriNodeList�еı�־, ����
	List<size_t, 1> d_olargeNodeTriNodeFlagsListAssist_;	//size_t		*d_olargeNodeTriNodeFlagsListAssist_;	// large node��newTriNodeList�еı�־, ���

	List<size_t, 1> d_ismallNodeFlagsListAssist_;			//size_t		*d_ismallNodeFlagsListAssist_;			// small node��nextNodeList�еı�־, ����
	List<size_t, 1> d_osmallNodeFlagsListAssist_;			//size_t		*d_osmallNodeFlagsListAssist_;			// small node��nextNodeList�еı�־, ���
	List<size_t, 1> d_ismallNodeNumListAssist_;			//size_t		*d_ismallNodeNumListAssist_;			// small node��nextNodeNumListAssist�еı�־������
	List<size_t, 1> d_osmallNodeNumListAssist_;			//size_t		*d_osmallNodeNumListAssist_;			// small node��nextNodeNumListAssist�еı�־�����
	List<size_t, 1> d_ismallNodeTriNodeFlagsListAssist_;	//size_t		*d_ismallNodeTriNodeFlagsListAssist_;	// small node��newTriNodeList�еı�־, ����
	List<size_t, 1> d_osmallNodeTriNodeFlagsListAssist_;	//size_t		*d_osmallNodeTriNodeFlagsListAssist_;	// small node��newTriNodeList�еı�־, ���



	//// temp
	size_t	*d_numValid_;		// ����compact��1�ĸ���, device mem
	size_t  *h_numValid_;		// ����compact, 1�ĸ���, host   mem


	/**************small node handle***************/
	List<int, 1> smallNodeList_; //KdNodeOrigin **smallNodeList_;			// small node
	//size_t       smallNodeListSize_;		// ��С
	List<size_t, 1> leafPrims_; //size_t       *leafPrims_;		// ����Ҷ�ӽڵ���������prims
	//size_t       leafSize_;
	size_t       leafNum_;
	size_t       smallProcTime_;

	List<int, 1> smallNodeRoot_;	//KdNodeOrigin **smallNodeRoot_;			// ԭʼ��small root
	List<size_t, 1> d_ismallNodeRootList_;		//size_t *d_ismallNodeRootList_;			// split����small node root�б�, ����
	List<size_t, 1> d_osmallNodeRootList_;		//size_t *d_osmallNodeRootList_;			// split����small node root�б�, ���	

	List<size_t, 1> smallNodeRootMaskHi_;		//size_t *smallNodeRootMaskHi_;			// ԭʼ��small root��mask�ĸ߶���Чλ
	List<size_t, 1> smallNodeRootMaskLo_;		//size_t *smallNodeRootMaskLo_;			// ԭʼ��small root��mask�ĵͶ���Чλ

	size_t smallNodeBoundrySize_;			// ��ѡsplit����Ŀ
	List<float, 3> smallNodeBoundryValue_;  //float  *smallNodeBoundryValue_[3];		// �߽�ֵ x, y, z, Ϊ�˽���segsort, �˹������˼�ֵ
	List<size_t, 3> smallNodeBoundryRPos_;  //size_t *smallNodeBoundryRPos_[3];		// boundry�����segflagλ��
	List<size_t, 1> smallNodeBoundryFlags_;    //size_t *smallNodeBoundryFlags_;			// boundry��flag, ��ʼ��λ1, ����Ϊ0
	List<size_t, 3> smallNodeBoundryType_;  //size_t *smallNodeBoundryType_[3];		// boundry������, ��ʼ��λ��Ϊ1, ����λ0
	List<size_t, 3> smallNodeBoundryTriIdx_;//size_t *smallNodeBoundryTriIdx_[3];		// boundry������tri
	List<size_t, 3> smallNodeBoundryAPos_;  //size_t *smallNodeBoundryAPos_[3];		// boundry�ľ���λ��, ����

	// ԭʼsmall node�б���ÿһ��node�а�����tris����Ŀ��scan֮������������е�ַ����
	List<size_t, 1> d_ismallNodeEveryLeafSize_; 
	List<size_t, 1> d_osmallNodeEveryLeafSize_; //size_t *d_osmallNodeEveryLeafSize_;
	
	// ??????????
	List<size_t, 1> d_ismallNodeSegStartAddr_;  //size_t *d_ismallNodeSegStartAddr_;
	List<size_t, 1> d_osmallNodeSegStartAddr_;  //size_t *d_osmallNodeSegStartAddr_;

	List<size_t, 3> d_ismallNodeMaskLeftHi_; //size_t *d_ismallNodeMaskLeftHi_[3];		// split���prims��Ŀ, ��λ, ����
	List<size_t, 3> d_osmallNodeMaskLeftHi_; //size_t *d_osmallNodeMaskLeftHi_[3];		// split���prims��Ŀ, ��λ, ���
	List<size_t, 3> d_ismallNodeMaskLeftLo_; //size_t *d_ismallNodeMaskLeftLo_[3];		// split���prims��Ŀ, ��λ, ����
	List<size_t, 3> d_osmallNodeMaskLeftLo_; //size_t *d_osmallNodeMaskLeftLo_[3];		// split���prims��Ŀ, ��λ, ��� 

	List<size_t, 3> d_ismallNodeMaskRightHi_; //size_t *d_ismallNodeMaskRightHi_[3];	// split�ұ�prims��Ŀ, ��λ, ����
	List<size_t, 3> d_osmallNodeMaskRightHi_; //size_t *d_osmallNodeMaskRightHi_[3];	// split�ұ�prims��Ŀ, ��λ, ���
	List<size_t, 3> d_ismallNodeMaskRightLo_; //size_t *d_ismallNodeMaskRightLo_[3];	// split�ұ�prims��Ŀ, ��λ, ����
	List<size_t, 3> d_osmallNodeMaskRightLo_; //size_t *d_osmallNodeMaskRightLo_[3];	// split�ұ�prims��Ŀ, ��λ, ��� 

	size_t smallNodeBoundryCapacity_;
	List<size_t, 1> smallNodeMaskHi_; //size_t *smallNodeMaskHi_;				// small node mask hi
	List<size_t, 1> smallNodeMaskLo_; //size_t *smallNodeMaskLo_;				// small node mask lo

	List<size_t, 1> smallNodeNextListMaskHi_; //size_t *smallNodeNextListMaskHi_;
	List<size_t, 1> smallNodeNextListMaskLo_; //size_t *smallNodeNextListMaskLo_;

	List<size_t, 1> smallNodeRList_;   //size_t *smallNodeRList_;

	List<size_t, 1> smallNodeNextListRList_; //size_t *smallNodeNextListRList_;

	List<size_t, 1> d_ismallNodeLeafFlags_; //size_t *d_ismallNodeLeafFlags_;			// ��Ҷ�ӽڵ���Ϊ1, ����, ������d_ismallNodeLeafSize_ compact��isValid,
	List<size_t, 1> d_osmallNodeLeafFlags_; //size_t *d_osmallNodeLeafFlags_;			// ��Ҷ�ӽڵ���Ϊ1, ���, ����scan ADD֮��, �õ�d_osmallNodeLeafSize_��leaf�Ĵ�С��λ��

	List<size_t, 1> d_ismallNodeNoLeafFlags_; //size_t *d_ismallNodeNoLeafFlags_;		// �ѷ�Ҷ�ӽڵ���Ϊ1, ����
	List<size_t, 1> d_osmallNodeNoLeafFlags_; //size_t *d_osmallNodeNoLeafFlags_;		// �ѷ�Ҷ�ӽڵ���Ϊ1, ���

	List<size_t, 1> d_ismallNodeLeafSize_; //size_t *d_ismallNodeLeafSize_;			// Ҷ�ӽڵ��к���prim����Ŀ, compact֮ǰ
	List<size_t, 1> d_osmallNodeLeafSize_; //size_t *d_osmallNodeLeafSize_;			// Ҷ�ӽڵ��к���prim����Ŀ, compact֮��, ���ڵõ�Ҷ�ӽڵ���������prims����ʼ��ַ


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
	//// ����
	Tex* tex_;
	int ppmNum_;
	void refTexture(Tex *tex, int ppmNum);
	void GPU_BuildKdTree::bindTexture2D(uchar4* h_gtex, uint4* h_texPos);
	void GPU_BuildKdTree::testTexture(Tex* tex, int ppmNum);
#endif

	//// ��ʼ��
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



	//// kd tree ����
	void buildTree();

	//// list ����
	template <typename T, int dimension>
    void swapList(List<T, dimension> &list1, List<T, dimension> &list2);

	//// ��activeList��ӵ�nodeList
	void largeNodeAppendNodeList();

	//// ��largeNodeList��active���н���
	void largeNodeSwap();

	//// ��largeNode�����е�һЩ�ڴ������չ(��̬��)
	void largeNodeExtendMem();

	//// �������ڵ�
	void createRoot();

	//// ����prims�İ�Χ��
	void calPrimsBoundingBox();

	//// ����large node
	void processLargeNode();

	//// �ռ�triNodeList������node��BB,����triNodePrimsBBList��ȥ��
	void collectTriNodePrimsBB();

	//// ����activeList��ÿ��node�İ�Χ��,��triNodePrimsBBList��Ϊdata,triNodeList:segFlags��Ϊflags, ִ��segmented scan
	void calActiveNodeBoundingBox();

	//// ����activeNodeList������node��boundingbox, splittype, splitValue,Ϊdistribute��׼��
	void set_ActiveNode_Child_PreDistribute();

	//// ��activeNodeList������node��splitvalue, splittype�����䵽����list��
	void distributeSplit();

	//// ��split����tri-node������tri���������ӻ����Һ���
	void setFlagWithSplit();

	//// ���㽫��nextList��ÿ��node�к���prim����Ŀ
	void calNextListNodePrimNum();

	//// ��left/rightChildListFlags�еĴ���next list node��prims��Ŀ��ֵ�����������nextNodeNumList
	void extractPrimNumFromChildFlags();

	//// ��nextNodeNumList����exclusive scan���õ�nextNodeListNew��newTriNodeList�е���ʼ��ַ
	void calNewTriNodeListAddr();

	//// ��triNodeList��left/rightChildListFlag����compact, �õ�newTriNodeList:triIdx
	void collectTrisToNewTriNodeList();

	//// ����newTriNodeListAssist_:flag, ������nextList��newTriNodeListAssit_֮�����ϵ, ����segment���г�ʼ��
	void finishNextList();

	//// �Ƿ���large nodeʱ���ٵĿռ�
    void largeNodeRelease();

	/********** small node filter*************/
	//// ��nextNodeNumList�������Һ��ӽڵ��compact
	void filter_extractNumList();

	//// ��largeNodeNumList��smallNodeNumList����exclusive scan���õ�������triNode����ʼ��ַ
	void filter_calTriNodeAddr();

	//// ��nextList�е�largeNode��smallNode�ֱ����compact
	void filter_extractNextList();

	//// ��large/small node triNodeFlagsList��segment scan��large/small�ı�־λ1
	void filter_distributeTriNodeFlags();

	//// ��newTriNodeList�зֱ���ȡ������large/small�ڵ��triIdx
	void extractTriIdx();

	//// ��ɹ��ˣ���kdNodeָ���triNode���¶�λ
	void filter_finish();

	//// filterȫ����
	void filter();

	/********** small node handle **************/
	//// ��small node����Ԥ����
	void preProcessSmallNode();

	//// ��small node������node����ԭʼ��mask
	void smallNode_createOriginMask();

	//// ��smallNodeBoundry���г�ʼ��
	void smallNode_initBound();

	//// ��smallNodeBoundryValue�ڸ������Ͻ���sorting
	void smallNode_boundrySorting();

	//// ��boundry�ĺ�ѡsplit��mask���г�ʼ����������seg scan(or and)
	void smallNode_setBoundryMask();

	//// ��ʼ������small node��һЩ����, ����ʽ����small node֮ǰ
	void smallNode_initProcessSmallNode();

	//// ��leaf node Ԥ����
	void smallNode_leafNodeFilter();

	//// ����split
	void smallNode_split();

	//// ��ʽ����small node���д���
	void processSmallNode();

	//// ��smallNodeList��smallNodeNextList���н���
	void smallNodeSwap();

	//// ��smallNode�����е�һЩ�ڴ������չ(��̬��)
	void smallNodeExtendMem();

	//// ��smallNodeList��ӵ�nodeList��
	void smallNodeAppendNodeList();

	//// �ͷŴ���small node���Դ���Դ
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
	
	// ���߸���
	// ��������
	Scene_d sceneParam_;
	uchar4 *d_pixelBuf_;
	uchar4 *h_pixelBuf_;
	uchar4* devPixelBufPtr_;
	void rayTrace();
    



};


#endif