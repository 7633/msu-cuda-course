#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <process.h> 

#include "../Common/HelloWorld.h"
#include "../Common/Rand.h"

int GetDeviceInfo(SYSTEM_INFO & SystemInfo, bool bPrintProp)
{

    GetSystemInfo( &SystemInfo ); 

    if (bPrintProp)
    {
        switch (SystemInfo.wProcessorArchitecture)
        {
            case PROCESSOR_ARCHITECTURE_INTEL:
                fprintf(stdout, "Processor architecture : Intel\n");
                switch (SystemInfo.wProcessorLevel)
                {
                    case 4:
                        fprintf(stdout, "Processor level : Intel 80486\n");
                        break;
                    case 5:
                        fprintf(stdout, "Processor level : Pentium\n");
                        break;
                    case 6:
                        fprintf(stdout, "Processor level : Intel Core 2\n");
                        break;
                    default:
                        fprintf(stdout, "Processor level : other\n");
                        break;
                }
                break;
            case PROCESSOR_ARCHITECTURE_AMD64:
                fprintf(stdout, "Processor architecture : Amd64\n");
                break;
            default:
                fprintf(stdout, "Processor architecture : other\n");
                break;
        }
        fprintf(stdout, "Number of processors : %d\n", SystemInfo.dwNumberOfProcessors);
        fprintf(stdout, "Current thread ID    : %d\n", GetCurrentThreadId());

#ifdef VISTA // GetCurrentProcessorNumber() is only available on Vista 
        fprintf(stdout, "Current thread runs on core # %d\n", GetCurrentProcessorNumber());
#endif
    }

    return SystemInfo.dwNumberOfProcessors;
}

int main ( int argc, char *  argv [] )
{
    SYSTEM_INFO systemInfo;
    
    int core_count = GetDeviceInfo( systemInfo, true );

    float * pA = NULL;
    float * pB = NULL;
    float * pC = NULL; // this will be used to read back data from GPU
    float * pD = NULL; // this will be used to store CPU Results

    int nMemoryInBytes = 1024 * 512;
    int nFloatElem = nMemoryInBytes / 4;
    int nFloatElemPerCore = nFloatElem / core_count;

    // allocate 4 arrays of 512 Kb each : 
    pA = (float *) malloc( nMemoryInBytes );
    pB = (float *) malloc( nMemoryInBytes );
    pC = (float *) malloc( nMemoryInBytes );
    pD = (float *) malloc( nMemoryInBytes );

    Rand(pA, nFloatElem, 2); // fill array A with random numbers
    Rand(pB, nFloatElem, 3); // fill array B with random numbers

    unsigned int start, stop;

    SArgList argList;
    argList.pA = pA;
    argList.pB = pB;
    argList.pC = pC;
    argList.start = 0;
    argList.stop = nFloatElem;
    argList.times = 1000;

    start = GetTickCount();

    MT_SimpleAddKernel((void *)&argList);

    stop = GetTickCount();

    float singleThreadTime = (stop - start);
    fprintf(stdout, "Single     thread time : %.5f  millseconds\n", singleThreadTime);

    HANDLE * pThreadHandle = (HANDLE *) malloc(core_count * sizeof(HANDLE));
    SArgList * pArgList = (SArgList *) malloc(core_count * sizeof(SArgList));

    pThreadHandle[0] = GetCurrentThread();

    pArgList[core_count-1] = argList;
    pArgList[core_count-1].pC = pD;
    pArgList[core_count-1].start = nFloatElem - nFloatElemPerCore;
    pArgList[core_count-1].stop = nFloatElem;

    // the job is divided between all possible threads
    for (int icore = core_count; icore > 1; icore--)
    {
        pThreadHandle[icore-1] = (HANDLE) _beginthread(MT_SimpleAddKernel, 0, (void *)&pArgList[icore-1]);
        
        pArgList[icore-2] = pArgList[icore-1];
        pArgList[icore-2].stop = pArgList[icore-1].start;
        pArgList[icore-2].start -= nFloatElemPerCore;
    }

    pArgList[0].start = 0;

    start = GetTickCount();

    // main thread does the rest of the job
    MT_SimpleAddKernel((void *)&pArgList[0]);

    // wait for the second thread to finish execution
    WaitForMultipleObjects(core_count-1, &pThreadHandle[1], true, INFINITE);

    stop = GetTickCount();

    float multiThreadTime = (stop - start);
    fprintf(stdout, "Multi (%d) thread time : %.5f  millseconds\n", core_count, multiThreadTime);

    // perform error checking
    for (int idx = 0; idx < nFloatElem; idx++)
    {
        if (pC[idx] != pD[idx])
        {
            fprintf(stdout, "error %d\n", idx);
            break;
        }
    }

    // free cpu resources
    free( pA );
    free( pB );
    free( pC );
    free( pD );

    free( pThreadHandle );
    free( pArgList );

    return 0;
}