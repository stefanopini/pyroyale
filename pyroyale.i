%module pyroyale

%header %{
/* Put header files here or function declarations like below */
#define SWIG_FILE_WITH_INIT
#define ROYALE_C_API_VERSION 31000

#include "pyroyale_const.h"
#include "pyroyale_internals.h"
#include "pyroyale.h"
%}


%init %{
/* Code to initialize the Python GIL and avoid the "Fatal Python error: take_gil: NULL tstate" error */
if (!PyEval_ThreadsInitialized()) {
    PyEval_InitThreads();

	/* Code to initialize the Numpy C-API library (avoiding the segmentation fault on PyArray_ContiguousFromObject) */
    if(PyArray_API == NULL) {
        import_array(); 
    }
}
%}


%include "pyroyale_const.h"
%include "pyroyale.h"
