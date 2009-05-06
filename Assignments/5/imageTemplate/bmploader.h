#ifndef _BMP_LOADER_H_
#define _BMP_LOADER_H_

extern "C"
bool LoadBMPFile(void **dst, int *width, int *height, const char *name);

#endif