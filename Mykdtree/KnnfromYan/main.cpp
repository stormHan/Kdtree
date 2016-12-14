// test.cpp : Defines the entry point for the console application.
//

#include <string>
#include <windows.h>
#include <time.h>
#include <istream>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>



#include <stdlib.h>
#include <stdio.h>
#include <vector>
//#include "./include/Triangle.h"
//#include "ParserModel.h"
//#include "cuda_gl_interop.h"
//#include "cuda.h"
//#include "dynamic_scene.h"

//#include "../RayTrace/RayTraceInterface.h"

/* 导入Cuda版射线追踪代码 */
//#include <CudaCode/CudaRayTrace.h>
//#include "../CudaRayTraceInterface.h"

#include "kdgpu_app.h"
#include "vector_types.h"

//#include "png.h"
//#include "zlib.h"
#include "png_image.h"

#include <iostream>
#include <iomanip>


#include <Gl/glew.h>
#include <GL/glut.h>
#include "cuda.h"
#include "cuda_gl_interop.h"


using namespace std;


//void genCoord(double x, double y, double z, double step, CBezierSet::BezierPatch *bp)
//{
//	bp->pCtrlPts = new VectorR3[9];
//	bp->pCtrlPts[0].x = x;
//	bp->pCtrlPts[0].y = y;
//	bp->pCtrlPts[0].z = z;
//
//	bp->pCtrlPts[1].x = x;
//	bp->pCtrlPts[1].y = y + step;
//	bp->pCtrlPts[1].z = z + step;
//
//	bp->pCtrlPts[2].x = x;
//	bp->pCtrlPts[2].y = y + 2 * step;
//	bp->pCtrlPts[2].z = z;
//
//	bp->pCtrlPts[3].x = x + step;
//	bp->pCtrlPts[3].y = y;
//	bp->pCtrlPts[3].z = z;
//
//	bp->pCtrlPts[4].x = x + step;
//	bp->pCtrlPts[4].y = y + step;
//	bp->pCtrlPts[4].z = z + step;
//
//	bp->pCtrlPts[5].x = x + step;
//	bp->pCtrlPts[5].y = y + 2 * step;
//	bp->pCtrlPts[5].z = z;
//
//	bp->pCtrlPts[6].x = x + 2 * step;
//	bp->pCtrlPts[6].y = y;
//	bp->pCtrlPts[6].z = z;
//
//	bp->pCtrlPts[7].x = x + 2 * step;
//	bp->pCtrlPts[7].y = y + step;
//	bp->pCtrlPts[7].z = z + step;
//
//	bp->pCtrlPts[8].x = x + 2 * step;
//	bp->pCtrlPts[8].y = y + 2 * step;
//	bp->pCtrlPts[8].z = z;
//
//
//}
//
// 生成测试用例
//void genTestCase(int n, CBezierSet &bs)
//{
//	// 分配空间
//    bs.m_nBPs = n;
//	bs.m_pBPs = new CBezierSet::BezierPatch[bs.m_nBPs];
//
//	// 生成
//	double step = 2.0;
//	
//	genCoord(0.0, 0.0, 0.0, step, bs.m_pBPs);
//	genCoord(30.0, 90.0, 0.0, step, bs.m_pBPs + 1);
//	genCoord(98.0, 80.0, 0.0, step, bs.m_pBPs + 2);
//	genCoord(196.0, 116.0, 0.0, step, bs.m_pBPs + 3);
//}
//
//
// 随机生成三角面片场景
// 最低点是世界坐标系原点
Tri *randomGenScene(int boxLen, int triLen, int n)
{
	srand((unsigned int)time(NULL));

	Tri *tris = (Tri *)malloc(sizeof(Tri) * n * 3);

	int num = boxLen / triLen;
	int x, y, z;
	int x1, y1, z1;
	for (int i = 0; i < n; i++)
	{
		x = (rand() % num) * triLen;
		y = (rand() % num) * triLen;
		z = (rand() % num) * triLen;

		for (int j = 0; j < 3; j++)
		{
			x1 = x + (rand() % (triLen + 1));
			y1 = y + (rand() % (triLen + 1));
			z1 = z + (rand() % (triLen + 1));

			tris[3 * i + j].x = ((float)x1) / 100;
			tris[3 * i + j].y = ((float)y1) / 100;
			tris[3 * i + j].z = ((float)z1) / 100;
		}



	}

	return tris;
}

// 把三角面片场景保存到文件中(obj)
void saveSceneToObj(Tri *tris, int n, char *filename)
{
	FILE *fp;
	fp = fopen(filename, "w+");
	//fprintf(fp, "%d\n", n);

	for (int i = 0; i < n; i++)
	{
		fprintf(fp, "v %f %f %f\n", tris[3 * i].x, tris[3 * i].y, tris[3 * i].z);
		fprintf(fp, "v %f %f %f\n", tris[3 * i + 1].x, tris[3 * i + 1].y, tris[3 * i + 1].z);
		fprintf(fp, "v %f %f %f\n", tris[3 * i + 2].x, tris[3 * i + 2].y, tris[3 * i + 2].z);
	}

	/*for (int i = 0; i < n; i++)
	{
	fprintf(fp, "f %d %d %d\n", 3 * i + 1, 3 * i + 2, 3 * i + 3);
	}*/

	fclose(fp);
}

bool isNum(char str)
{
	return str >= '0' && str <= '9';
}

int getIndex(char *str, int *vertexIndex, int *textureIndex, int *normalIndex)
{
	int num;
	int ret = 0;
	int indexState = -1; // 0为读取顶点索引，1为读取纹理索引，2为读取顶点法向索引
	int spaceState = 0; // 0为空格状态，1为读取索引状态

	int vertexCount = 0;
	int textureCount = 0;
	int normalCount = 0;

	vertexIndex[0] = -1;
	textureIndex[0] = -1;
	normalIndex[0] = -1;

	int i = 0;
	while (1)
	{
		if (isNum(str[i]) && spaceState == 0)
		{
			spaceState = 1;
			indexState++;

			ret++;
		}
		if (str[i] == ' ' && spaceState == 1)
		{
			switch (indexState)
			{
			case 0:
				vertexIndex[vertexCount++] = num;
				break;
			case 1:
				textureIndex[textureCount++] = num;
				break;
			case 2:
				normalIndex[normalCount++] = num;
			default:
				break;
			}
			spaceState = 0;
			indexState = -1;
		}

		if (spaceState == 0) num = 0;

		if (str[i] >= '0' && str[i] <= '9')
		{
			num = num * 10 + str[i] - '0';
		}
		else if (str[i] == '/')
		{
			switch (indexState)
			{
			case 0:
				vertexIndex[vertexCount++] = num;
				break;
			case 1:
				textureIndex[textureCount++] = num;
				break;
			case 2:
				normalIndex[normalCount++] = num;
			default:
				break;
			}

			num = 0;
			indexState++;
		}
		else if (str[i] == '\0')
		{
			switch (indexState)
			{
			case 0:
				vertexIndex[vertexCount++] = num;
				break;
			case 1:
				textureIndex[textureCount++] = num;
				break;
			case 2:
				normalIndex[normalCount++] = num;
			default:
				break;
			}

			break;
		}
		/*else if(str[i] = ' ')
		{
		spaceState = 0;
		}*/
		i++;
	}

	for (int i = 0; i < vertexCount; i++)
	{
		if (vertexIndex[i] < 0 || textureIndex[i] < 0 || normalIndex[i] < 0)
			printf("negtive value!\n");
	}

	return ret;
}

void saveSceneToObj(SceneInfo &scene, char *filename)
{
	FILE *fp;
	fp = fopen(filename, "w+");
	for (int i = 0; i < scene.vertexNum; i++)
	{
		fprintf(fp, "v %f %f %f\n", scene.h_vertex[i].x, scene.h_vertex[i].y, scene.h_vertex[i].z);
	}
	/*for (int i = 0; i < scene.faceNum; i++)
	{
	fprintf(fp, "f %d %d %d\n", scene.h_face[i].x + 1, scene.h_face[i].y + 1, scene.h_face[i].z + 1);
	}*/
	fclose(fp);
}

// 把随机生成的场景保存到模型中
void saveScene(Tri *tris, int n, char *filename)
{
	FILE *fp;
	fp = fopen(filename, "w+");

	fprintf(fp, "%d\n", n);
	for (int i = 0; i < n; i++)
	{
		fprintf(fp, "%f %f %f\n", tris[3 * i].x, tris[3 * i].y, tris[3 * i].z);
		fprintf(fp, "%f %f %f\n", tris[3 * i + 1].x, tris[3 * i + 1].y, tris[3 * i + 1].z);
		fprintf(fp, "%f %f %f\n", tris[3 * i + 2].x, tris[3 * i + 2].y, tris[3 * i + 2].z);
	}
	fclose(fp);
}

// 从场景中读取三角面片模型
Tri *loadScene(int &n, char *filename)
{
	FILE *fp;

	fp = fopen(filename, "r+");
	fscanf(fp, "%d", &n);
	Tri *tris = (Tri *)malloc(sizeof(Tri) * 3 * n);
	for (int i = 0; i < n; i++)
	{
		fscanf(fp, "%f %f %f\n", &tris[3 * i].x, &tris[3 * i].y, &tris[3 * i].z);
		fscanf(fp, "%f %f %f\n", &tris[3 * i + 1].x, &tris[3 * i + 1].y, &tris[3 * i + 1].z);
		fscanf(fp, "%f %f %f\n", &tris[3 * i + 2].x, &tris[3 * i + 2].y, &tris[3 * i + 2].z);
	}

	fclose(fp);
	return tris;
}


void objtooff(int vNum, int fNum, VertexType* ptrV, FaceType* ptrF, char* path)
{
	FILE *fpoff;
	//char offpath[300] = "E:\\kd construction result\\Toasters\\Toasters004.off";
	fpoff = fopen(path, "w+");

	// OFF
	fprintf(fpoff, "off\n");

	// vNum fNum
	fprintf(fpoff, "%d %d\n", vNum, fNum);

	// vertex points
	for (int i = 0; i < vNum; i++)
	{
		fprintf(fpoff, "%f %f %f\n", ptrV[i].x, ptrV[i].y, ptrV[i].z);
	}

	// face
	for (int i = 0; i < fNum; i++)
	{
		fprintf(fpoff, "3 %d %d %d\n", ptrF[i].x, ptrF[i].y, ptrF[i].z);
	}

	fclose(fpoff);

}
/////////////// opengl ////////////
int hei = 800;
int wid = 800;

float rotateSpeed = 3.1415926;
float wheelSpeed = 0.1;
float tranSpeed = 1.0;//314.15926; 

FLOAT3 bbMin = { FLOAT_MAX, FLOAT_MAX, FLOAT_MAX };
FLOAT3 bbMax = { FLOAT_MIN, FLOAT_MIN, FLOAT_MIN };
FLOAT3 bsCenter = { 0.305318 / 2, 0.707192 / 2, 0.521658 / 2 };
GLfloat bsRadius = 1.414;

FLOAT3 defLookPos = { 415.374207, 675.03, 454.638 };
FLOAT3 defLookAt = bsCenter;
FLOAT3 defLookUp = { 0.0, 75.903, 0.0 };

FLOAT3 defPointLightColor = { 1.0, 1.0, 1.0 };
FLOAT3 defPointLightPos = defLookPos;

FLOAT3 lookPos = defLookPos;
FLOAT3 lookAt = defLookAt;
FLOAT3 lookUp = defLookUp;

FLOAT defNear = 0.01;
FLOAT defFar = FLOAT_MAX;

FLOAT defFovy = 3.1415926 / 2;
FLOAT defAspect = 1.0;

int prevX = 0;
int prevY = 0;

bool isCameraRotate = false; // 照相机旋转
bool isCameraTran = false;   // 照相机平移

Scene_d sceneParam;
GPU_BuildKdTree *kdTree;
float3* pixelBuf;

// opengl and cuda
GLuint  bufferObj;
cudaGraphicsResource *resource;
uchar4* devPixelBufPtr;

// timer
unsigned int timer = 0;

// 帧率
float ifps;

// key frame
int keyframe = 0;
int keynum;


void init(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);


	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void computeFPS()
{
	char fps[256];
	ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
	sprintf(fps, "speed: %.1f fps", ifps);
	//int n = strlen(fps);
	//设置要在屏幕上显示字符的起始位置
	//glRasterPos2i(100,100);
	//逐个显示字符串中的每个字符
	/*for (int i = 0; i < n; i++)
	glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *(fps+i));*/

	glutSetWindowTitle(fps);
	cutResetTimer(timer);
}

void display(void)
{
	// start timer
	cutStartTimer(timer);

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0); // 准备在屏幕(0,0)，左下角处开始绘制图像

	// do work with the memory dst being on the GPU, gotten via mapping
	// 每次对显存操作时，都要重新映射一次
	cudaGraphicsMapResources(1, &resource, NULL);
	size_t  size;
	cudaGraphicsResourceGetMappedPointer((void**)&devPixelBufPtr,
		&size,
		resource);

	// 更新device pixel pointer
	kdTree->devPixelBufPtr_ = devPixelBufPtr;
	/*cudaError_t e = cudaGetLastError();
	printf("error code: %d\n", e);*/
	kdTree->rayTrace();
	/*e = cudaGetLastError();
	printf("error code: %d\n", e);*/

	// 每次用完都要去掉映射
	cudaGraphicsUnmapResources(1, &resource, NULL);

	// 绘制像素
	glDrawPixels(sceneParam.view_.wid_, sceneParam.view_.hei_,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);

	/*glDrawPixels( sceneParam.view_.wid_, sceneParam.view_.hei_,
	GL_RGBA, GL_UNSIGNED_BYTE, kdTree->h_pixelBuf_ );*/

	/*glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2f(0, 0);
	for (int i = 0; i < 90; i++)
	glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, (int)'a');*/


	// 双缓存
	glutSwapBuffers();
	glutPostRedisplay();

	// end timer
	cutStopTimer(timer);
	computeFPS();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	GLint height = (GLint)h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case 27:
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
		break;

	default:
		break;
	}

}

static void mouse_button(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:

		//printf("left button\n");
		if (GLUT_DOWN == state)
		{
			isCameraRotate = true;
			prevX = x;
			prevY = hei - y - 1;
		}
		else
		{
			isCameraRotate = false;
		}

		char fps[256];
		sprintf(fps, "speed: %.1f fps, pos: (%d, %d)", ifps, x, y);
		glutSetWindowTitle(fps);


		break;
	case GLUT_MIDDLE_BUTTON:
		lookPos = defLookPos;
		lookAt = defLookAt;
		//display();
		//glutPostRedisplay();
		break;
	case GLUT_RIGHT_BUTTON:
		if (GLUT_DOWN == state)
		{
			isCameraTran = true;
			prevX = x;
			prevY = hei - y - 1;
		}
		else
		{
			isCameraRotate = false;
			//lookAt = defLookAt;
		}
		break;
	//case GLUT_WHEEL_UP:
	//{
	//	FLOAT3 vq = sub3(bsCenter, lookPos);
	//	float vqLen = len3(vq);
	//	nor3(vq);

	//	float step = 1.0 * wheelSpeed * vqLen;
	//	if (step >= vqLen) step = 0.0;
	//	FLOAT3 oq = multi3(vq, vqLen - step);
	//	lookPos = sub3(bsCenter, oq);

	//	kdTree->sceneParam_.camera_.from_ = float3to4(lookPos);
	//	kdTree->sceneParam_.pointLight_.pos_ = float3to4(lookPos);


	//}
	//printf("up\n");



	break;
	//case GLUT_WHEEL_DOWN:
	//	//printf("down\n");
	//{
	//	FLOAT3 vq = sub3(bsCenter, lookPos);
	//	float vqLen = len3(vq);
	//	nor3(vq);
	//	float step = 1.0 * wheelSpeed * vqLen;
	//	FLOAT3 oq = multi3(vq, vqLen + step);
	//	lookPos = sub3(bsCenter, oq);

	//	kdTree->sceneParam_.camera_.from_ = float3to4(lookPos);
	//	kdTree->sceneParam_.pointLight_.pos_ = float3to4(lookPos);
	//}
	//break;
	}

	/*printf("pos: %f %f %f\n", lookPos.x, lookPos.y, lookPos.z);
	printf("at: %f %f %f\n", bsCenter.x, bsCenter.y, bsCenter.z);
	printf("at: %f %f %f\n", lookUp.x, lookUp.y, lookUp.z);*/
	glutPostRedisplay();
}

static void mouse_motion(int x, int y)
{
	// 照相机旋转
	if (isCameraRotate)
	{
		int curX = x;
		int curY = hei - y - 1;

		FLOAT3 vq = sub3(bsCenter, lookPos);
		float vqLen = len3(vq);

		int rangeX = prevX - curX;
		int rangeY = prevY - curY;

		GLfloat dirX = vqLen * rotateSpeed * rangeX * 2.0 / wid;
		GLfloat dirY = vqLen * rotateSpeed * rangeY * 2.0 / hei;

		prevX = curX;
		prevY = curY;

		nor3(vq);
		FLOAT3 oq = multi3(vq, vqLen - 1 / tan(3.1415926 / 4));

		FLOAT3 axisX = cross3(vq, lookUp);
		nor3(axisX);
		FLOAT3 axisY = cross3(axisX, vq);
		nor3(axisY);

		FLOAT3 op = add3(multi3(axisX, dirX), multi3(axisY, dirY));
		FLOAT3 qp = sub3(op, oq);

		nor3(qp);
		FLOAT3 qv1 = multi3(qp, vqLen);
		FLOAT3 newPos = add3(bsCenter, qv1);
		lookPos = newPos;

		FLOAT3 qvnor = multi3(vq, -1);
		FLOAT3 r = cross3(qvnor, qp);
		nor3(r);


		if (abs(rangeX) > abs(rangeY))
		{
			FLOAT3 a = axisX;
			FLOAT3 c = cross3(r, a);
			nor3(c);
			float cos_ = dot3(qvnor, qp);
			float sin_ = sqrt(1 - cos_ * cos_);
			FLOAT3 b = add3(multi3(a, cos_), multi3(c, sin_));

			FLOAT3 newLookUp = cross3(qp, b);
			nor3(newLookUp);
			lookUp = newLookUp;
		}
		else
		{
			FLOAT3 a = axisY;
			FLOAT3 c = cross3(r, a);
			nor3(c);
			float cos_ = dot3(qvnor, qp);
			float sin_ = sqrt(1 - cos_ * cos_);
			FLOAT3 b = add3(multi3(a, cos_), multi3(c, sin_));

			nor3(b);
			lookUp = b;
			/*FLOAT3 newLookUp = cross(qp, b);
			nor(newLookUp);
			lookUp = newLookUp;*/
		}

		kdTree->sceneParam_.camera_.from_ = float3to4(lookPos);
		kdTree->sceneParam_.camera_.up_ = float3to4(lookUp);
		kdTree->sceneParam_.camera_.to_ = float3to4(bsCenter);
		kdTree->sceneParam_.camera_.calCamera();
		kdTree->sceneParam_.pointLight_.pos_ = float3to4(lookPos);

		/*printf("pos: %f %f %f\n", lookPos.x, lookPos.y, lookPos.z);
		printf("at: %f %f %f\n", bsCenter.x, bsCenter.y, bsCenter.z);
		printf("up: %f %f %f\n", lookUp.x, lookUp.y, lookUp.z);*/



		//display();

	}

	// 照相机平移
	if (isCameraTran)
	{
		int curX = x;
		int curY = hei - y - 1;

		FLOAT3 vq = sub3(bsCenter, lookPos);
		float vqLen = len3(vq);

		int rangeX = prevX - curX;
		int rangeY = prevY - curY;


		GLfloat dirX = tranSpeed * rangeX * 2.0 / wid;
		GLfloat dirY = tranSpeed * rangeY * 2.0 / hei;

		prevX = curX;
		prevY = curY;

		nor3(vq);
		FLOAT3 oq = multi3(vq, vqLen - 1 / tan(3.1415926 / 4));

		FLOAT3 axisX = cross3(vq, lookUp);
		nor3(axisX);
		FLOAT3 axisY = cross3(axisX, vq);
		nor3(axisY);

		FLOAT3 op = add3(multi3(axisX, dirX), multi3(axisY, dirY));
		lookPos = add3(lookPos, op);
		bsCenter = lookAt = add3(lookAt, op);

		kdTree->sceneParam_.camera_.from_ = float3to4(lookPos);
		kdTree->sceneParam_.camera_.up_ = float3to4(lookUp);
		kdTree->sceneParam_.camera_.to_ = float3to4(lookAt);
		kdTree->sceneParam_.camera_.calCamera();
		kdTree->sceneParam_.pointLight_.pos_ = float3to4(lookPos);

		//display();
		//glFlush();

	}



	glutPostRedisplay();



}

unsigned int BKDRHash(char *str)
{
	unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
	unsigned int hash = 0;

	while (*str)
	{
		hash = hash * seed + (*str++);
	}

	return (hash & 0x7FFFFFFF);
}

// 返回不重复的ppm的数目
int excludeRepeatPPM(vector<Mtl>&mtlLib, Tex *tex)
{
	// 遍历所有的ppm
	int index = 0;
	int i, j;
	for (i = 0; i < mtlLib.size(); i++)
	{
		// 只对有ppm文件的材质读取
		if (!mtlLib[i].isMapKd) continue;
		string iMapkd = mtlLib[i].map_Kd;

		for (j = 0; j < i; j++)
		{
			if (!mtlLib[i].isMapKd) continue;
			string jMapkd = mtlLib[j].map_Kd;

			if (iMapkd == jMapkd)
			{
				mtlLib[i].texIndex = mtlLib[j].texIndex;
				break;
			}
		}

		if (j == i)
		{
			mtlLib[i].texIndex = index;

			tex[index].wid = mtlLib[i].wid;
			tex[index].hei = mtlLib[i].hei;
			memcpy(tex[index].map_Kd, mtlLib[i].map_Kd, 100);
			tex[index].texPtr = NULL;

			index++;
		}
	}

	return index;
}



void loadMtlLib(char *scenePath, char *mtlPath, int pathLen, vector<Mtl> &mtlLib, Tex *tex, int &ppmNum,
	uchar4* &h_gtex, uint4* &h_texPos)
{
	char tempStr[400];
	char prefix[50];
	char mtlName[50];

	char *mtlRealPath = new char[pathLen];
	memcpy(mtlRealPath, scenePath, pathLen);
	strcat(mtlRealPath, mtlPath);

	printf("---loading mtllib: %s\n", mtlRealPath);


	// 读取mtllib信息
	ifstream inputMtl;
	inputMtl.open(mtlRealPath);
	delete[] mtlRealPath;

	string temp;
	Mtl newMtl;
	int libSize = 0;
	bool isMap = false;

	while (!inputMtl.eof())
	{
		inputMtl.getline(tempStr, 500, '\n');

		int readNum = sscanf(tempStr, "%s", prefix);
		if (readNum < 0)
		{
			continue;
		}
		temp = prefix;
		if (temp == "newmtl")
		{
			if (libSize != 0)
			{
				newMtl.isMapKd = isMap;
				newMtl.texIndex = -1;
				mtlLib.push_back(newMtl);
				isMap = false;
			}



			libSize++;
			sscanf(tempStr, "%s %s", prefix, mtlName);
			newMtl.name = mtlName;
		}
		else if (temp == "Ka")
		{
			sscanf(tempStr, "%s %f %f %f", prefix, &newMtl.Ka.x, &newMtl.Ka.y, &newMtl.Ka.z);
		}
		else if (temp == "Kd")
		{
			sscanf(tempStr, "%s %f %f %f", prefix, &newMtl.Kd.x, &newMtl.Kd.y, &newMtl.Kd.z);
		}
		else if (temp == "Ks")
		{
			sscanf(tempStr, "%s %f %f %f", prefix, &newMtl.Ks.x, &newMtl.Ks.y, &newMtl.Ks.z);
		}
		else if (temp == "Tr" || temp == "d")
		{
			sscanf(tempStr, "%s %f", prefix, &newMtl.Tr);
		}
		else if (temp == "Ns")
		{
			sscanf(tempStr, "%s %f", prefix, &newMtl.Ns);
		}
		else if (temp == "map_Kd")
		{
			sscanf(tempStr, "%s %s", prefix, newMtl.map_Kd);
			isMap = true;
		}
	}
	newMtl.isMapKd = isMap;
	newMtl.texIndex = -1;
	mtlLib.push_back(newMtl);
	inputMtl.close();

	// ppm 文件去重
	ppmNum = excludeRepeatPPM(mtlLib, tex);

	// 读取像素信息(ppm, bmp, jpg, png)
	h_gtex = new uchar4[TEX_G_WID * TEX_G_HEI];
	h_texPos = new uint4[TEXMAX];

	h_texPos[0].x = TEX_G_WID;
	h_texPos[0].y = TEX_G_HEI;

	int starth = 0;
	for (int p = 0; p < ppmNum; p++)
	{

		// 根据文件名后缀，判断是文件什么类型的像素文件
		char *strPtr = tex[p].map_Kd;
		int len = strlen(strPtr);
		char suffix[10];
		string suffixName;
		int j = 0;
		for (int i = len - 1; i >= 0; i--)
		{
			char str = strPtr[i];
			if (str == '.') break;
			suffix[j++] = strPtr[i];
		}
		suffix[j] = '\0';
		char c;
		for (int i = 0; i < j / 2; i++)
		{
			c = suffix[i];
			suffix[i] = suffix[j - i - 1];
			suffix[j - i - 1] = c;
		}
		suffixName = suffix;
		if (suffixName == "ppm") // ppm
		{
			ifstream inputTex;
			char *ppmPath = new char[pathLen];
			memcpy(ppmPath, scenePath, pathLen);
			strcat(ppmPath, strPtr);

			printf("---loading PPM: %s\n", ppmPath);

			// open
			inputTex.open(ppmPath);


			//// read
			bool isP3;

			// type
			inputTex.getline(tempStr, 500, '\n');
			int readNum = sscanf(tempStr, "%s", prefix);
			temp = prefix;
			if (temp == "P3")   // P3: ASCII
			{
				isP3 = true;
			}
			else				// P6: Binary
			{
				isP3 = false;
			}

			// width & height
			int lwid, lhei;
			inputTex.getline(tempStr, 500, '\n');
			inputTex.getline(tempStr, 500, '\n');
			sscanf(tempStr, "%d %d", &lwid, &lhei);


			h_texPos[p + 1].x = lwid;
			h_texPos[p + 1].y = lhei;
			h_texPos[p + 1].z = starth;
			// max value
			int maxPixelValue;
			inputTex.getline(tempStr, 500, '\n');
			sscanf(tempStr, "%d", &maxPixelValue);

			int pixelNum = 0;
			if (isP3)
			{

				for (int h = 0; h < lhei; h++)
				{
					for (int w = 0; w < lwid; w++)
					{
						uint3 pixelInfo;

						inputTex.getline(tempStr, 500, '\n');
						sscanf(tempStr, "%d", &pixelInfo.x);
						inputTex.getline(tempStr, 500, '\n');
						sscanf(tempStr, "%d", &pixelInfo.y);
						inputTex.getline(tempStr, 500, '\n');
						sscanf(tempStr, "%d", &pixelInfo.z);

						int index = (starth + h) * TEX_G_WID + w;
						h_gtex[index].x = pixelInfo.x;
						h_gtex[index].y = pixelInfo.y;
						h_gtex[index].z = pixelInfo.z;
						h_gtex[index].w = 255;
					}
				}
				starth += lhei;
			}
			else
			{
				for (int h = 0; h < lhei; h++)
				{
					for (int w = 0; w < lwid; w++)
					{
						uchar3 pixel;

						inputTex.read((char *)&pixel, sizeof(uchar3));

						int index = (starth + h) * TEX_G_WID + w;
						h_gtex[index].x = pixel.x;
						h_gtex[index].y = pixel.y;
						h_gtex[index].z = pixel.z;
						h_gtex[index].w = 255;
					}
				}

				starth += lhei;
			}

			inputTex.close();
			delete[] ppmPath;

		}
		else if (suffixName == "png" || suffixName == "PNG")
		{
			char *ppmPath = new char[pathLen];
			memcpy(ppmPath, scenePath, pathLen);
			strcat(ppmPath, strPtr);
			printf("---loading PNG: %s\n", ppmPath);

			int lwid, lhei;
			unsigned char* rgba = readpng(ppmPath, lwid, lhei);
			uchar4* pixels = (uchar4 *)rgba;
			h_texPos[p + 1].x = lwid;
			h_texPos[p + 1].y = lhei;
			h_texPos[p + 1].z = starth;

			for (int h = 0; h < lhei; h++)
			{
				for (int w = 0; w < lwid; w++)
				{
					int gindex = (starth + h) * TEX_G_WID + w;
					int lindex = h * lwid + w;
					h_gtex[gindex].x = pixels[lindex].x;
					h_gtex[gindex].y = pixels[lindex].y;
					h_gtex[gindex].z = pixels[lindex].z;
					h_gtex[gindex].w = 255;
				}
			}
			delete[] rgba;
			starth += lhei;
		}
	}

	//// new format
	//fclose(fileB);

}

int findMtlIndex(vector<Mtl>&mtlLib, string mtlNameStr)
{
	vector<Mtl>::const_iterator iter;
	//int index = 0;
	for (iter = mtlLib.begin(); iter != mtlLib.end(); iter++)
	{
		if ((*iter).name == mtlNameStr && (*iter).isMapKd)
			//return index;
			return (*iter).texIndex;
		//index++;
	}

	// if failed
	return -1;
}

void loadDynamicModel(SceneInfoArr &sceneArr, int &ppmNum,
	uchar4* &h_gtex, uint4* &h_texPos)
{

	//// choice dynamic model
	const int pathLen = 300;
	int maxprim = 300000;
	char scenePath[pathLen] = "E:/xialong/dynamic/binary/";
AGAIN:
	printf("select model:\n");
	printf("1.Ben\n");
	printf("2.Wooddoll\n");
	printf("3.Toaster\n");
	printf("4.Fairyforest\n");
	printf("5.Marble\n");
	int select = 4;
	char objName[100];
	scanf("%d", &select);
	switch (select)
	{
	case 1:
		strcpy(objName, "ben.dyn");
		maxprim = 108000;
		break;
	case 2:
		strcpy(objName, "wooddoll.dyn");
		maxprim = 12000;
		break;
	case 3:
		strcpy(objName, "toaster.dyn");
		maxprim = 79000;
		break;
	case 4:
		strcpy(objName, "fairy.dyn");
		maxprim = 175000;
		break;
	default:
		goto AGAIN;
		break;
	}

	char objPath[pathLen];
	memcpy(objPath, scenePath, pathLen);
	strcat(objPath, objName);
	printf("---loading model: %s\n", objPath);

	//// init variable

	size_t vertexNum[KEY_MAX];
	size_t faceNum[KEY_MAX];

	size_t normalNum[KEY_MAX];
	size_t faceNormalNum[KEY_MAX];

	size_t textureNum[KEY_MAX];
	size_t faceTextureNum[KEY_MAX];

	VertexType *h_vertex[KEY_MAX];
	FaceType *h_face[KEY_MAX];

	NormalType *h_normal[KEY_MAX];
	FaceVNType *h_facenormal[KEY_MAX];

	TextureType *h_texture[KEY_MAX];
	FaceVTType *h_facetexture[KEY_MAX];

	//// read mtl
	FILE *fp = fopen(objPath, "rb");

	// ppm num
	printf("loading ppm……\n");
	fread(&ppmNum, sizeof(int), 1, fp);

	// wid, hei
	int lwid, lhei;
	int starth = 0;

	h_gtex = new uchar4[TEX_G_WID * TEX_G_HEI];
	h_texPos = new uint4[TEXMAX];

	h_texPos[0].x = TEX_G_WID;
	h_texPos[0].y = TEX_G_HEI;

	for (int i = 0; i < ppmNum; i++)
	{
		fread(&lwid, sizeof(int), 1, fp);
		fread(&lhei, sizeof(int), 1, fp);

		h_texPos[i + 1].x = lwid;
		h_texPos[i + 1].y = lhei;
		h_texPos[i + 1].z = starth;

		for (int h = 0; h < lhei; h++)
		{
			int index = (starth + h) * TEX_G_WID;
			fread(h_gtex + index, sizeof(uchar4) * lwid, 1, fp);

		}
		starth += lhei;
	}

	//// read geometry
	// key num
	printf("loading keyframe……\n");
	fread(&keynum, sizeof(int), 1, fp);



	// geo
	int steplen = 1024; // 2^10
	int binarylen = 10;
	int t, r;
	for (int i = 0; i < keynum; i++)
	{
		printf("loading keyframe: %d……\n", i);
		// alloc mem
		h_vertex[i] = (VertexType *)malloc(sizeof(VertexType) * maxprim);
		h_face[i] = (FaceType *)malloc(sizeof(FaceType) * maxprim);

		h_normal[i] = (NormalType *)malloc(sizeof(NormalType) * maxprim);
		h_facenormal[i] = (FaceVNType *)malloc(sizeof(FaceVNType) * maxprim);

		h_texture[i] = (TextureType *)malloc(sizeof(TextureType) * maxprim);
		h_facetexture[i] = (FaceVTType *)malloc(sizeof(FaceVTType) * maxprim);

		int j;
		// vertex
		fread(vertexNum + i, sizeof(int), 1, fp);
		t = vertexNum[i] >> binarylen;
		r = vertexNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_vertex[i] + (j << binarylen), steplen * sizeof(VertexType), 1, fp);
		fread(h_vertex[i] + (t << binarylen), r * sizeof(VertexType), 1, fp);

		// bb
		if (i == 0)
		{
			for (int v = 0; v < vertexNum[0]; v++)
			{
				bbMin.x = (h_vertex[0][v].x < bbMin.x) ? h_vertex[0][v].x : bbMin.x;
				bbMax.x = (h_vertex[0][v].x > bbMax.x) ? h_vertex[0][v].x : bbMax.x;
				bbMin.y = (h_vertex[0][v].y < bbMin.y) ? h_vertex[0][v].y : bbMin.y;
				bbMax.y = (h_vertex[0][v].y > bbMax.y) ? h_vertex[0][v].y : bbMax.y;
				bbMin.z = (h_vertex[0][v].z < bbMin.z) ? h_vertex[0][v].z : bbMin.z;
				bbMax.z = (h_vertex[0][v].z > bbMax.z) ? h_vertex[0][v].z : bbMax.z;
			}
		}


		// normal
		fread(normalNum + i, sizeof(int), 1, fp);
		t = normalNum[i] >> binarylen;
		r = normalNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_normal[i] + (j << binarylen), steplen * sizeof(NormalType), 1, fp);
		fread(h_normal[i] + (t << binarylen), r * sizeof(NormalType), 1, fp);

		// texture
		fread(textureNum + i, sizeof(int), 1, fp);
		t = textureNum[i] >> binarylen;
		r = textureNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_texture[i] + (j << binarylen), steplen * sizeof(TextureType), 1, fp);
		fread(h_texture[i] + (t << binarylen), r * sizeof(TextureType), 1, fp);

		// face
		fread(faceNum + i, sizeof(int), 1, fp);
		t = faceNum[i] >> binarylen;
		r = faceNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_face[i] + (j << binarylen), steplen * sizeof(FaceType), 1, fp);
		fread(h_face[i] + (t << binarylen), r * sizeof(FaceType), 1, fp);

		// face normal
		fread(faceNormalNum + i, sizeof(int), 1, fp);
		t = faceNormalNum[i] >> binarylen;
		r = faceNormalNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_facenormal[i] + (j << binarylen), steplen * sizeof(FaceVNType), 1, fp);
		fread(h_facenormal[i] + (t << binarylen), r * sizeof(FaceVNType), 1, fp);

		// face normal
		fread(faceTextureNum + i, sizeof(int), 1, fp);
		t = faceTextureNum[i] >> binarylen;
		r = faceTextureNum[i] - (t << binarylen);
		for (j = 0; j < t; j++)
			fread(h_facetexture[i] + (j << binarylen), steplen * sizeof(FaceVTType), 1, fp);
		fread(h_facetexture[i] + (t << binarylen), r * sizeof(FaceVTType), 1, fp);

		sceneArr.vertexNum[i] = vertexNum[i];
		sceneArr.normalNum[i] = normalNum[i];
		sceneArr.textureNum[i] = textureNum[i];
		sceneArr.faceNum[i] = faceNum[i];
		sceneArr.faceNormalNum[i] = faceNormalNum[i];
		sceneArr.faceTextureNum[i] = faceTextureNum[i];
		sceneArr.h_vertex[i] = h_vertex[i];
		sceneArr.h_normal[i] = h_normal[i];
		sceneArr.h_texture[i] = h_texture[i];
		sceneArr.h_face[i] = h_face[i];
		sceneArr.h_facenormal[i] = h_facenormal[i];
		sceneArr.h_facetexture[i] = h_facetexture[i];
	}




	/*VertexType *h_vertex = (VertexType *)malloc(sizeof(VertexType) * maxlen);
	FaceType *h_face = (FaceType *)malloc(sizeof(FaceType) * maxlen);

	NormalType *h_normal = (NormalType *)malloc(sizeof(NormalType) * maxlen);
	FaceVNType *h_facenormal = (FaceVNType *)malloc(sizeof(FaceVNType) * maxlen);

	TextureType *h_texture = (TextureType *)malloc(sizeof(TextureType) * maxlen);
	FaceVTType *h_facetexture = (FaceVTType *)malloc(sizeof(FaceVTType) * maxlen);*/



}


void loadModel_obj(SceneInfo& sceneInfo, vector<Mtl> &mtlLib, Tex *tex, int &ppmNum,
	uchar4* &h_gtex, uint4* &h_texPos)
{
	const int pathLen = 300;
	// 设置模型路径
	char scenePath[pathLen] = "E:/xialong/dynamic/kd construction result/dynamicmodels/";

	//  char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\ben_00_origin.obj";
	//char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\wooddoll_00_origin.obj";
	//char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\Toasters_modify.obj";
	//char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\marbles000_orgin.obj";
	//char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\fairyforest_origin.obj";
	//char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\asd4.obj";
AGAIN:
	printf("select model:\n");
	printf("1.Ben\n");
	printf("2.Wooddoll\n");
	printf("3.Toaster\n");
	printf("4.Fairyforest\n");
	printf("5.Marble\n");
	int select = 4;
	char objName[100];
	scanf("%d", &select);
	switch (select)
	{
	case 1:
		strcpy(objName, "ben_00_origin.obj");
		break;
	case 2:
		strcpy(objName, "wooddoll_00_origin.obj");
		break;
	case 3:
		strcpy(objName, "Toasters_modify.obj");
		break;
	case 4:
		strcpy(objName, "fairyforest_origin.obj");
		break;
	case 5:
		strcpy(objName, "marbles000_orgin.obj");
		break;
	default:
		goto AGAIN;
		//exit(0);
		break;
	}

	char objPath[pathLen];
	memcpy(objPath, scenePath, pathLen);
	strcat(objPath, objName);
	printf("---loading model: %s\n", objPath);

	int vertexNum = 0;
	int faceNum = 0;

#ifdef NEED_NORMAL
	int normalNum = 0;
	int faceNormalNum = 0;
#endif

#ifdef NEED_TEXTURE
	int textureNum = 0;
	int faceTextureNum = 0;
#endif

	VertexType *h_vertex = (VertexType *)malloc(sizeof(VertexType) * 300000);
	FaceType *h_face = (FaceType *)malloc(sizeof(FaceType) * 300000);

#ifdef NEED_NORMAL
	NormalType *h_normal = (NormalType *)malloc(sizeof(NormalType) * 300000);
	FaceVNType *h_facenormal = (FaceVNType *)malloc(sizeof(FaceVNType) * 300000);
#endif

#ifdef NEED_TEXTURE
	TextureType *h_texture = (TextureType *)malloc(sizeof(TextureType) * 300000);
	FaceVTType *h_facetexture = (FaceVTType *)malloc(sizeof(FaceVTType) * 300000);
#endif

	// mtl
	//vector<Mtl> mtlLib;
	bool isUseMtlLib = false;
	int mtlIndex = -1; // 默认没有纹理


	ifstream input;
	char tempStr[400];
	char prefix[50];
	std::string temp;


	int vertexIndex[4];
	int textureIndex[4];
	int normalIndex[4];
	//int index[12];
	int pointNum;

	int readLines = 0;



	input.open(objPath);
	while (!input.eof())
	{
		input.getline(tempStr, 500, '\n');
		readLines++;
		//printf("line: %d, content: %s\n", readLines, tempStr);
		int readNum = sscanf(tempStr, "%s", prefix);
		if (readNum < 0)
		{
			continue;
		}
		temp = prefix;
		if (temp == "v")
		{
			// 读取顶点信息
			sscanf(tempStr, "%s %f %f %f", prefix,
				&h_vertex[vertexNum].x, &h_vertex[vertexNum].y, &h_vertex[vertexNum].z);


			bbMin.x = (h_vertex[vertexNum].x < bbMin.x) ? h_vertex[vertexNum].x : bbMin.x;
			bbMax.x = (h_vertex[vertexNum].x > bbMax.x) ? h_vertex[vertexNum].x : bbMax.x;
			bbMin.y = (h_vertex[vertexNum].y < bbMin.y) ? h_vertex[vertexNum].y : bbMin.y;
			bbMax.y = (h_vertex[vertexNum].y > bbMax.y) ? h_vertex[vertexNum].y : bbMax.y;
			bbMin.z = (h_vertex[vertexNum].z < bbMin.z) ? h_vertex[vertexNum].z : bbMin.z;
			bbMax.z = (h_vertex[vertexNum].z > bbMax.z) ? h_vertex[vertexNum].z : bbMax.z;

			vertexNum++;
		}
#ifdef NEED_TEXTURE
		else if (temp == "mtllib") // 材质库读取
		{
			char mtlPath[50];
			sscanf(tempStr, "%s %s", prefix, mtlPath);
			loadMtlLib(scenePath, mtlPath, pathLen, mtlLib, tex, ppmNum, h_gtex, h_texPos);
		}
		else if (temp == "usemtl")
		{
			isUseMtlLib = true;
			char mtlName[50];
			sscanf(tempStr, "%s %s", prefix, mtlName);
			string mtlNameStr = mtlName;
			// find mtl index
			mtlIndex = findMtlIndex(mtlLib, mtlNameStr);
		}
#endif
		else if (temp == "f")
		{
			// 读取面信息,可能是3个顶点，可能是4个顶点
			sscanf(tempStr, "%s ", prefix); // 读取前缀

			pointNum = getIndex(tempStr, vertexIndex, textureIndex, normalIndex);
			if (3 == pointNum)
			{
				// face
				h_face[faceNum].x = vertexIndex[0] - 1;
				h_face[faceNum].y = vertexIndex[1] - 1;
				h_face[faceNum].z = vertexIndex[2] - 1;
				h_face[faceNum].w = mtlIndex;
				faceNum++;

#ifdef NEED_TEXTURE
				// texture
				h_facetexture[faceTextureNum].x = textureIndex[0] - 1;
				h_facetexture[faceTextureNum].y = textureIndex[1] - 1;
				h_facetexture[faceTextureNum].z = textureIndex[2] - 1;
				faceTextureNum++;
#endif

#ifdef NEED_NORMAL
				// normal
				h_facenormal[faceNormalNum].x = normalIndex[0] - 1;
				h_facenormal[faceNormalNum].y = normalIndex[1] - 1;
				h_facenormal[faceNormalNum].z = normalIndex[2] - 1;
				faceNormalNum++;
#endif
			}
			else if (4 == pointNum)
			{
				// face 
				h_face[faceNum].x = vertexIndex[0] - 1;
				h_face[faceNum].y = vertexIndex[1] - 1;
				h_face[faceNum].z = vertexIndex[2] - 1;
				h_face[faceNum].w = mtlIndex;
				faceNum++;
				h_face[faceNum].x = vertexIndex[0] - 1;// 0 2 3
				h_face[faceNum].y = vertexIndex[2] - 1;
				h_face[faceNum].z = vertexIndex[3] - 1;
				h_face[faceNum].w = mtlIndex;
				faceNum++;

#ifdef NEED_TEXTURE
				// texture
				h_facetexture[faceTextureNum].x = textureIndex[0] - 1;
				h_facetexture[faceTextureNum].y = textureIndex[1] - 1;
				h_facetexture[faceTextureNum].z = textureIndex[2] - 1;
				faceTextureNum++;
				h_facetexture[faceTextureNum].x = textureIndex[0] - 1;
				h_facetexture[faceTextureNum].y = textureIndex[2] - 1;
				h_facetexture[faceTextureNum].z = textureIndex[3] - 1;
				faceTextureNum++;
#endif 

#ifdef NEED_NORMAL
				// normal
				h_facenormal[faceNormalNum].x = normalIndex[0] - 1;
				h_facenormal[faceNormalNum].y = normalIndex[1] - 1;
				h_facenormal[faceNormalNum].z = normalIndex[2] - 1;
				faceNormalNum++;

				h_facenormal[faceNormalNum].x = normalIndex[0] - 1;// 0 2 3
				h_facenormal[faceNormalNum].y = normalIndex[2] - 1;
				h_facenormal[faceNormalNum].z = normalIndex[3] - 1;
				faceNormalNum++;
#endif
			}
			else
			{
				printf("read obj error!\n");
				return;
			}
		}

#ifdef NEED_TEXTURE
		else if (temp == "vt") // 纹理坐标
		{
			sscanf(tempStr, "%s %f %f %f", prefix,
				&h_texture[textureNum].x, &h_texture[textureNum].y, &h_texture[textureNum].z);
			textureNum++;
		}
#endif

#ifdef NEED_NORMAL
		else if (temp == "vn") // 顶点法向量
		{
			// 读取顶点信息
			sscanf(tempStr, "%s %f %f %f", prefix,
				&h_normal[normalNum].x, &h_normal[normalNum].y, &h_normal[normalNum].z);
			normalNum++;
		}
#endif
	}



	input.close();

	sceneInfo.vertexNum = vertexNum;
	sceneInfo.faceNum = faceNum;
	sceneInfo.h_vertex = h_vertex;
	sceneInfo.h_face = h_face;

#ifdef NEED_TEXTURE
	sceneInfo.textureNum = textureNum;
	sceneInfo.faceTextureNum = faceTextureNum;
	sceneInfo.h_texture = h_texture;
	sceneInfo.h_facetexture = h_facetexture;
#endif

#ifdef NEED_NORMAL
	sceneInfo.normalNum = normalNum;
	sceneInfo.faceNormalNum = faceNormalNum;
	sceneInfo.h_normal = h_normal;
	sceneInfo.h_facenormal = h_facenormal;
#endif
}


void setSceneParam()
{
	bsCenter.x = (bbMax.x + bbMin.x) / 2;
	bsCenter.y = (bbMax.y + bbMin.y) / 2;
	bsCenter.z = (bbMax.z + bbMin.z) / 2;

	lookAt = defLookAt = bsCenter;

	// 3.11
	float3 newLookDir, newLookPos;
	newLookDir.x = bbMax.x - bsCenter.x;
	newLookDir.y = bbMax.y - bsCenter.y;
	newLookDir.z = bbMax.z - bsCenter.z;
	newLookDir.x *= 1.5;
	newLookDir.y *= 1.5;
	newLookDir.z *= 1.5;
	newLookPos.x = bsCenter.x + newLookDir.x;
	newLookPos.y = bsCenter.y + newLookDir.y;
	newLookPos.z = bsCenter.z + newLookDir.z;
	lookPos.f3 = defLookPos.f3 = newLookPos;
	defPointLightPos = defLookPos;

	/*defPointLightPos.x = lookPos.x = defLookPos.x = 415.374207;
	defPointLightPos.y = lookPos.y = defLookPos.y = 675.03;
	defPointLightPos.z = lookPos.z = defLookPos.z = 454.638;

	lookAt.x = defLookAt.x = 0.0;
	lookAt.y = defLookAt.y = 75.903;
	lookAt.z = defLookAt.z = 0.0;*/

	/*fairy defPointLightPos.x = lookPos.x = defLookPos.x = 2.667412;
	defPointLightPos.y = lookPos.y = defLookPos.y = 1.909359;
	defPointLightPos.z = lookPos.z = defLookPos.z = 1.342083;

	lookAt.x = defLookAt.x = -0.016209;
	lookAt.y = defLookAt.y = 0.740527;
	lookAt.z = defLookAt.z = 0.708130;

	defLookUp.x = -0.382943;
	defLookUp.y = 0.920615;
	defLookUp.z = -0.076299;*/




	// light
	FLOAT4 lightPos;
	FLOAT4 lightColor;

	//lightPos.x = 1.0f; // light pos
	//lightPos.y = 1.0f;
	//lightPos.z = 1.0f;
	//lightPos.w = 1.0f;
	lightPos = float3to4(defPointLightPos);

	//lightColor.x = 1.0f;	// light color
	//lightColor.y = 1.0f;
	//lightColor.z = 1.0f;
	lightColor = float3to4(defPointLightColor);

	sceneParam.pointLight_.pos_ = lightPos;
	sceneParam.pointLight_.color_ = lightColor;

	// camera 
	//Camera_d camera;
	FLOAT4 from;
	FLOAT4 to;
	FLOAT4 up;
	FLOAT near1;
	FLOAT far1;
	FLOAT fovy;
	FLOAT aspect;

	/*from.x = 0.5f;
	from.y = 0.0f;
	from.z = 0.0f;*/
	int rx = bbMax.x - bbMin.x;
	int ry = bbMax.y - bbMin.y;
	int rz = bbMax.z - bbMin.z;
	int maxr = (rx > rz) ? ((rx > ry) ? rx : ry) : ((rz > ry) ? rz : ry);
	//defLookPos.x = maxr * 1.5;
	from = float3to4(defLookPos);

	/*to.x = 0.3f;
	to.y = 0.2f;
	to.z = -1.8f;*/
	to = float3to4(defLookAt);

	printf("pos: %f %f %f\n", defLookPos.x, defLookPos.y, defLookPos.z);
	printf("at: %f %f %f\n", defLookAt.x, defLookAt.y, defLookAt.z);


	/*up.x = 0.0f;
	up.y = 1.0f;
	up.z = 0.0f;*/
	up = float3to4(defLookUp);

	near1 = defNear;
	far1 = defFar;

	fovy = defFovy;
	aspect = defAspect;

	sceneParam.camera_.from_ = from;
	sceneParam.camera_.to_ = to;
	sceneParam.camera_.up_ = up;
	sceneParam.camera_.near_ = near1;
	sceneParam.camera_.far_ = far1;
	sceneParam.camera_.fovy_ = fovy;
	sceneParam.camera_.aspect_ = aspect;
	sceneParam.camera_.calCamera();

	// view
	sceneParam.view_.wid_ = wid;
	sceneParam.view_.hei_ = hei;
	sceneParam.view_.samplePerPixel_ = 1;
}

// convert obj to pbrt
void obj2pbrt(SceneInfo &scene)
{
	FILE *fp;
	/*
	strcpy(objName, "ben_00_origin.obj");
	break;
	case 2:
	strcpy(objName, "wooddoll_00_origin.obj");
	break;
	case 3:
	strcpy(objName, "Toasters_modify.obj");
	break;
	case 4:
	strcpy(objName, "fairyforest_origin.obj");
	break;
	case 5:
	strcpy(objName, "marbles000_orgin.obj");
	*/
	char* path = "c:/fairy.pbrt";
	fp = fopen(path, "w+");
	fprintf(fp, "Shape \"trianglemesh\"\n");
	fprintf(fp, "\"integer indices\"\n");
	fprintf(fp, "[\n");
	// face
	for (int i = 0; i < scene.faceNum; i++)
	{
		fprintf(fp, "%d %d %d\n", scene.h_face[i].x,
			scene.h_face[i].y, scene.h_face[i].z);
	}
	fprintf(fp, "]\n");
	fprintf(fp, "\"point P\"\n");
	fprintf(fp, "[\n");
	// vertex
	for (int i = 0; i < scene.vertexNum; i++)
	{
		fprintf(fp, "%f %f %f\n", scene.h_vertex[i].x,
			scene.h_vertex[i].y, scene.h_vertex[i].z);
	}
	fprintf(fp, "]\n");


	fclose(fp);
}

void renderScene(void)
{
	// release prev kd-tree
	kdTree->kdNodeBase_.release();
	kdTree->kdNodeExtra_.release();
	kdTree->kdNodeBB_.release();

	kdTree->leafPrims_.release();

	kdTree->keyframe_ = keyframe;
	kdTree->initialization();
	kdTree->buildTree();
	printf("keynum: %d", keynum);
	printf("keyframe: %d\n", keyframe);
	keyframe = (keyframe + 1) % keynum;

	glutSwapBuffers();
}

int main(int argc, char* argv[])
{
	// read obj
	char scenepath[300] = "E:\\kd construction result\\dynamicmodels\\ben_00_origin.obj";
	char objpath[300] = "E:\\kd construction result\\dynamicmodels\\wooddoll_00";

	//// create timer
	cutCreateTimer(&timer);

	//// load scene
	SceneInfo scene;
	vector<Mtl>mtlLib;
	Tex *tex = new Tex[TEXMAX];
	int ppmNum;
	uchar4* h_gtex;
	uint4* h_texPos;

	SceneInfoArr sceneArr;
	uchar4* h_gtexArr;
	uint4* h_texPosArr;
	int ppmNumArr;
	loadDynamicModel(sceneArr, ppmNum, h_gtex, h_texPos);
	/*scene.vertexNum = sceneArr.vertexNum[15];
	scene.normalNum = sceneArr.normalNum[15];
	scene.textureNum = sceneArr.textureNum[15];
	scene.faceNum = sceneArr.faceNum[15];
	scene.faceNormalNum = sceneArr.faceNormalNum[15];
	scene.faceTextureNum = sceneArr.faceTextureNum[15];
	scene.h_vertex = sceneArr.h_vertex[15];
	scene.h_normal = sceneArr.h_normal[15];
	scene.h_texture = sceneArr.h_texture[15];
	scene.h_face = sceneArr.h_face[15];
	scene.h_facenormal = sceneArr.h_facenormal[15];
	scene.h_facetexture = sceneArr.h_facetexture[15];*/
	//exit(0);
	//loadModel_obj(scene, mtlLib, tex, ppmNum, h_gtex, h_texPos);

	// obj2pbrt
	// obj2pbrt(scene);

	//// set scene information
	setSceneParam();

	//// set opengl & CUDA
	cudaDeviceProp  prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	if (cudaSuccess == cudaChooseDevice(&dev, &prop))
		printf("cudaChooseDevice is successful\n");;
	cudaGLSetGLDevice(dev);

	// gl init
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(sceneParam.view_.wid_, sceneParam.view_.hei_);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Scharoun");
	init();

	// 必须初始化
	glewInit();

	// 建立缓冲区对象
	// the first three are standard OpenGL, the 4th is the CUDA reg 
	// of the bitmap these calls exist starting in OpenGL 1.5
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sceneParam.view_.wid_ * sceneParam.view_.hei_ * 4,
		NULL, GL_DYNAMIC_DRAW_ARB);

	//cudaError_t e = cudaGetLastError();

	cudaGraphicsGLRegisterBuffer(&resource,
		bufferObj,
		cudaGraphicsMapFlagsNone);
	//cudaError_t e = cudaGetLastError();

	//// build kd tree
	GPU_BuildKdTree newTree;
	kdTree = &newTree;
	newTree.globalInit(sceneArr, keynum, h_gtex, h_texPos);
	newTree.initialization();
	newTree.sceneParam_ = sceneParam;
	newTree.devPixelBufPtr_ = devPixelBufPtr;
	newTree.buildTree();

	//// gl callback funs
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse_button);
	glutMotionFunc(mouse_motion);
	glutIdleFunc(renderScene);
	glutMainLoop();


	//delete [] scenePath;
	//delete [] resultPath;


}

