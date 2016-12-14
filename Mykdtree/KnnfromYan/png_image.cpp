#include "png_image.h"
#include "string.h"
#include "malloc.h"
#include "png.h"

bool writepng(char* name, int width, int height, unsigned char* data)
{
	FILE* fp = fopen(name, "wb");
	if (!fp)
	{
		printf("open file %s error!\n", name);
		return false;
	}
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	// png_colorp palette;


    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }


	png_set_IHDR(png_ptr, info_ptr, width, height, BIT_DEPTH, PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

	/* Write the file header information.  REQUIRED */
	png_write_info(png_ptr, info_ptr);

	if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

	//png_bytep image = (png_bytep)malloc(height * width * BYTES_PER_PIXEL);
	//int bytesize = sizeof(png_byte);
	png_bytepp row_pointers = (png_bytepp)malloc(sizeof(png_bytep) * height);

	for (int k = 0; k < height; k++)
		row_pointers[k] = data + k * width * BYTES_PER_PIXEL;

	/*for (int i = 0; i < height * width; i++)
	{
		image[i * 4] = 255;
		image[i * 4 + 1] = 255;
		image[i * 4 + 2] = 0;
		image[i * 4 + 3] = 255;

	}*/

	png_write_image(png_ptr, row_pointers);

	if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);

	// free(image);
    fclose(fp);

	return true;
}


unsigned char* readpng(char* name, int& w,  int& h)
{

	// ǰ�߼����ǳ�������ʼ�����ֽṹ

	FILE* file = fopen(name, "rb");

	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);

	png_infop info_ptr = png_create_info_struct(png_ptr);

	setjmp(png_jmpbuf(png_ptr));

	// ������Ҫ

	png_init_io(png_ptr, file);

	// ���ļ���

	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND, 0);

	// �õ��ļ��Ŀ��ɫ��

	w = png_get_image_width(png_ptr, info_ptr);
	h = png_get_image_height(png_ptr, info_ptr);

	// int color_type = png_get_color_type(png_ptr, info_ptr);
	int pixelSize = png_get_rowbytes(png_ptr, info_ptr) / w;

	// ������ڴ����棬�����õ���c++�﷨��������c�ϱ��

	int size = h * w * 4;

	unsigned char* bgra = new unsigned char[size];

	int pos = 0;

	// row_pointers��߾��Ǵ�˵�е�rgba������

	png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

	// ��������ע�⣬������ȡ��pngû��Aͨ������Ҫ3λ3λ�Ķ������о���ע���ֽڶ�������⣬��򵥵ľ��Ǳ��ò��ܱ�4�����Ŀ�Ⱦ����ˡ�������ʵ�����ã���Ҫ�����������صĶ��봦��

	for(int i = 0; i < h; i++)
	{
	   for(int j = 0; j < (pixelSize * w); j += pixelSize)
	   {
			bgra[pos++] = row_pointers[i][j];   // red
			bgra[pos++] = row_pointers[i][j + 1]; // green
			bgra[pos++] = row_pointers[i][j + 2]; // blue
		
			if (pixelSize == 4)
				bgra[pos++] = row_pointers[i][j + 3]; // alpha
			else 
				pos++;

	   }
	}

	// ���ˣ������������������κε������ˡ�����������ʾ�������ߴ�ӡ�������С�

	png_destroy_read_struct(&png_ptr, &info_ptr, 0);

	fclose(file);
	return bgra;
}
