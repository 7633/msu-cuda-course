#include <stdio.h> 
#include <windows.h>
#include <process.h>   // для beginthread()

void mtPrintf( void * pArg); 

int main() 
{ 
    int t0 = 0; int t1 = 1;
	_beginthread(mtPrintf, 0, (void*)&t0 ); 

	mtPrintf( (void*)&t1); 

	Sleep( 100 ); 

	return 0;
} 
void mtPrintf( void * pArg ) 
{ 
	int * pIntArg = (int *) pArg;
	printf( "The function was passed %d\n", (pIntArg[0]) ); 
}
