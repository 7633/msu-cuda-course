#include <stdio.h>
#include <stdlib.h>

#include "bmploader.h"

#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(push)
#pragma pack(1)

typedef struct
{
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct
{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;


//Isolated definition
typedef struct
{
    unsigned char x, y, z, w;
} uchar4;

#pragma pack(pop)


extern "C" 
{
    bool LoadBMPFile(void **pDst, int *width, int *height, const char *name)
    {
        BMPHeader hdr;
        BMPInfoHeader infoHdr;
        int x, y;

        FILE *fd;

        printf("Loading %s...\n", name);
        if(sizeof(uchar4) != 4){
            printf("***Bad uchar4 size***\n");
            return false;
        }

        if( !(fd = fopen(name,"rb")) ){
            printf("***BMP load error: file access denied***\n");
            return false;
        }

        fread(&hdr, sizeof(hdr), 1, fd);
        if(hdr.type != 0x4D42){
            printf("***BMP load error: bad file format***\n");
            return false;
        }
        fread(&infoHdr, sizeof(infoHdr), 1, fd);

        if(infoHdr.bitsPerPixel != 24){
            printf("***BMP load error: invalid color depth***\n");
            return false;
        }

        if(infoHdr.compression){
            printf("***BMP load error: compressed image***\n");
            return false;
        }

        *width  = infoHdr.width;
        *height = infoHdr.height;

        uchar4 * dst = (uchar4 *)malloc(*width * *height * 4);

        printf("BMP width: %u\n", infoHdr.width);
        printf("BMP height: %u\n", infoHdr.height);

        fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

        for(y = 0; y < infoHdr.height; y++){
            for(x = 0; x < infoHdr.width; x++){
                dst[(y * infoHdr.width + x)].z = fgetc(fd);
                dst[(y * infoHdr.width + x)].y = fgetc(fd);
                dst[(y * infoHdr.width + x)].x = fgetc(fd);
            }

            for(x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
                fgetc(fd);
        }


        if(ferror(fd))
        {
            printf("***Unknown BMP load error.***\n");
            free(dst);
            return false;
        }else
            printf("BMP file loaded successfully!\n");

        *pDst = dst;

        fclose(fd);

        return true;
    }
}