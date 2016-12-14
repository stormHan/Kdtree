#ifndef _PNG_IMAGE_H_
#define _PNG_IMAGE_H_
#include "png.h"
#include "zlib.h"

#define BIT_DEPTH 8
#define BYTES_PER_PIXEL 4

// Éú³ÉpngÍ¼Ïñ
bool writepng(char* name, int width, int height, unsigned char* data);

// ¶ÁÈ¡pngÍ¼Ïñ
unsigned char* readpng(char* name, int& w, int& h);


#endif