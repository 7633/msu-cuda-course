#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "glew.h"
#include "glut.h"

GLuint g_Tex = 0;
GLuint g_PBO = 0;

GLuint g_W = 0;
GLuint g_H = 0;
GLuint g_Radius = 9;

#include "bmploader.h"
#include "Convolution.h"

unsigned int glCreateTexture(int w, int h, unsigned char * pData)
{
    unsigned int tex = 0;
    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, (GLuint *) &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pData);

    GLenum error = glGetError();
    if (error == GL_NO_ERROR) 
    {
        printf("Texture created.\n");
        g_W = w;
        g_H = h;
        return tex;
    }
    else 
    {
        printf("Error while creating texture!\n");
        return 0;
    }
}

unsigned int glCreatePBO(int w, int h)
{
    unsigned int pbo = 0;
    printf("Creating PBO...\n");
    glGenBuffers(1, (GLuint *) &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, NULL, GL_STREAM_COPY);
    GLenum error = glGetError();
    if (error == GL_NO_ERROR)
    {
        printf("PBO created.\n");
        return pbo;
    }
    else
    {
        printf("Error while creating PBO!\n");
        return 0;
    }
}

void Display(void)
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(2, 0); glVertex2f(+3, -1);
    glTexCoord2f(0, 2); glVertex2f(-1, +3);
    glEnd();

    Wrapper_Convolution_Run(g_Radius);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, g_W, g_H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glutSwapBuffers();
}

void Idle(void)
{
    Display();
}

void Keyboard(unsigned char key, int mousex, int mousey)
{
    switch (key)
    {
        case 'q': 
        case 'Q':
        case GLUT_KEY_ESC:
            Wrapper_Convolution_Release();
            glDeleteTextures(1, &g_Tex);
            glDeleteBuffers (1, &g_PBO);
            exit(0);
            break;

        default:
            break;
    }
}

void Reshape(int w, int h)
{

}

int main( int argc, char** argv) 
{

    unsigned char * pRGBA = NULL;
    int w, h;

    if ( !LoadBMPFile( (void **) &pRGBA, &w, &h, "portrait_noise.bmp") )
        return 1;

    glutInit( &argc, argv );    
    glutInitWindowSize(w, h);    
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutCreateWindow("image processing template");

    glutDisplayFunc(Display);
    glutIdleFunc(Idle);
    glutKeyboardFunc(Keyboard);
    glutReshapeFunc(Reshape);

    GLenum error = glewInit();
    if (error != GL_NO_ERROR)
        return -1;

    g_Tex = glCreateTexture(w, h, pRGBA);
    g_PBO = glCreatePBO(w, h);

    Wrapper_Convolution_Init( pRGBA, w, h, g_PBO );

    glutMainLoop();

    return 0;
}