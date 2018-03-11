#ifndef _PYROYALE_INTERNALS_H
#define _PYROYALE_INTERNALS_H

#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include <royale.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <memory>

#include <Python.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>



/// Helper functions for converting Python <-> C

/*
/// Convert a array of string into Python list of string
PyObject *to_list_of_strings(royale::Vector<royale::String> &strings);

/// Convert royale_pair_string_string into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, royale::String>> &pairs);

/// Convert royale_pair_string_double into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, double>> &pairs);

/// Convert royale_pair_string_int into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, long>> &pairs);

/// Convert royale_pair_string_size_t into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, size_t>> &pairs);

/// Convert a array of double into Python list of double
PyObject* to_list_of_double(royale::Vector<double> &elems);
*/

PyObject *to_PyOjbect(royale::String &var);

PyObject *to_PyOjbect(const char *var);

PyObject *to_PyOjbect(double var);

PyObject *to_PyOjbect(long var);

PyObject *to_PyOjbect(size_t var);

template<typename T>
PyObject *to_python_list(royale::Vector<T> &vect);

template<typename T>
PyObject *to_python_dict(royale::Vector<royale::Pair<royale::String, T>> &pairs);


/// Set Python error message from royale error status
void set_error_message(const royale::CameraStatus status, const char *message, PyObject *error_type = PyExc_BaseException);

/// Logging function
void logging(const char * format, ...);


////////////////////////////////////////////////////////////////////////////////


// Callbacks

//void call_python_callback(PyObject *callback, PyObject *x_array, PyObject *y_array, PyObject *z_array, PyObject *noise, PyObject *gray_array, PyObject *depth_confidence);
template<typename... Args>
void call_python_callback(PyObject* callback, Args... vars);

template<typename T>
PyObject* convert_buffer_to_numpy_array(const royale::Vector<T> &data, const royale::Vector<npy_intp> &dims, const NPY_TYPES type);


////////////////////////////////////////////////////////////////////////////////


#endif // !_PYROYALE_INTERNALS_H