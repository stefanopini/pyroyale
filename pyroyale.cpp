#define SWIG_FILE_WITH_INIT
//#define ROYALE_C_API_VERSION 31000
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include <royale.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <cmath>

#include <Python.h>

#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

#include "pyroyale_const.h"
#include "pyroyale_internals.h"
#include "pyroyale.h"

//using namespace royale;
//using namespace sample_utils;
//using namespace std;



/// helper definition in place of enum classes
// royale::ExposureMode
size_t EXPOSURE_MODE_MANUAL = 0;
size_t EXPOSURE_MODE_AUTO = 1;
// royale::CameraAccessLevel
size_t CAMERA_ACCESS_LEVEL_L1 = 1;
size_t CAMERA_ACCESS_LEVEL_L2 = 2;
size_t CAMERA_ACCESS_LEVEL_L3 = 3;
size_t CAMERA_ACCESS_LEVEL_L4 = 4;
// royale::EventSeverity
size_t EVENT_SECURITY_ROYALE_INFO = 0;
size_t EVENT_SECURITY_ROYALE_WARNING = 1;
size_t EVENT_SECURITY_ROYALE_ERROR = 2;
size_t EVENT_SECURITY_ROYALE_FATAL = 3;
// royale::EventType
size_t EVENT_TYPE_ROYALE_CAPTURE_STREAM = 0;
size_t EVENT_TYPE_ROYALE_DEVICE_DISCONNECTED = 1;
size_t EVENT_TYPE_ROYALE_OVER_TEMPERATURE = 2;
size_t EVENT_TYPE_ROYALE_RAW_FRAME_STATS = 3;



/// Helper functions for converting Python <-> C

/*
/// Convert a array of string into Python list of string
PyObject *to_list_of_strings(royale::Vector<royale::String> &strings) {
PyObject *ret = PyList_New(strings.size());
for (size_t i = 0; i < strings.size(); i++) {
PyList_SetItem(ret, i, to_PyOjbect(strings[i].data()));
}
return ret;
}

/// Convert royale_pair_string_string into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, royale::String>> &pairs) {
PyObject *ret = PyDict_New();
for (size_t i = 0; i < pairs.size(); ++i) {
PyDict_SetItemString(ret, pairs[i].first.data(), to_PyOjbect(pairs[i].second.data()));
}
return ret;
}

/// Convert royale_pair_string_double into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, double>> &pairs) {
PyObject *ret = PyDict_New();
for (size_t i = 0; i < pairs.size(); ++i) {
PyDict_SetItemString(ret, pairs[i].first.data(), to_PyOjbect(pairs[i].second));
}
return ret;
}

/// Convert royale_pair_string_int into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, long>> &pairs) {
PyObject *ret = PyDict_New();
for (size_t i = 0; i < pairs.size(); ++i) {
PyDict_SetItemString(ret, pairs[i].first.data(), PyLong_FromLong(pairs[i].second));
}
return ret;
}

/// Convert royale_pair_string_size_t into Python dict
PyObject *to_dict(royale::Vector<royale::Pair<royale::String, size_t>> &pairs) {
PyObject *ret = PyDict_New();
for (size_t i = 0; i < pairs.size(); ++i) {
PyDict_SetItemString(ret, pairs[i].first.data(), PyLong_FromSize_t(pairs[i].second));
}
return ret;
}

/// Convert a array of double into Python list of double
PyObject *to_list_of_double(royale::Vector<double> &elems) {
PyObject *ret = PyList_New(elems.size());
for (size_t i = 0; i < elems.size(); i++) {
PyList_SetItem(ret, i, PyFloat_FromDouble(elems[i]));
}
return ret;
}
*/

PyObject *to_PyOjbect(royale::String &var) {
	return PyBytes_FromString(var.data());
}

PyObject *to_PyOjbect(const char *var) {
	return PyBytes_FromString(var);
}

PyObject *to_PyOjbect(double var) {
	return PyFloat_FromDouble(var);
}

PyObject *to_PyOjbect(long var) {
	return PyLong_FromLong(var);
}

PyObject *to_PyOjbect(size_t var) {
	return PyLong_FromSize_t(var);
}

template<typename T>
PyObject *to_python_list(royale::Vector<T> &vect) {
	PyObject *ret = PyList_New(vect.size());
	for (size_t i = 0; i < vect.size(); i++) {
		PyList_SetItem(ret, i, to_PyOjbect(vect[i]));
	}
	return ret;
}

template<typename T>
PyObject *to_python_dict(royale::Vector<royale::Pair<royale::String, T>> &pairs) {
	PyObject *ret = PyDict_New();
	for (size_t i = 0; i < pairs.size(); ++i) {
		PyDict_SetItemString(ret, pairs[i].first.data(), to_PyOjbect(pairs[i].second));
	}
	return ret;
}


/// Set Python error message from royale error status
void set_error_message(royale::CameraStatus status, const char *message, PyObject *error_type/* = PyExc_BaseException*/) {
	//char *error_string = royale_status_get_error_string(status);
	PyErr_Format(error_type, "%s; %s", message, royale::getStatusString(status).data());
}

/// Logging function
void logging(const char *format, ...) {
#ifdef DEBUG
	va_list args;
	printf("[DEBUG]");
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
#endif
}


////////////////////////////////////////////////////////////////////////////////

// Camera Manager


//using cm = CameraManager;
PyCameraManager::PyCameraManager() : manager() {};
PyCameraManager::~PyCameraManager() {
	delete manager;
}

PyObject *PyCameraManager::initialize() {
	manager = new royale::CameraManager();
	//handle_ = royale_camera_manager_create();
	if (false) {
		PyErr_Format(PyExc_RuntimeError, "Failed to create camera manager");
		return NULL;
	}
	else {
		Py_RETURN_NONE;
	}
}

PyObject *PyCameraManager::get_connected_cameras() const {
	royale::Vector<royale::String> cameras = manager->getConnectedCameraList();
	
	PyObject *ret = to_python_list(cameras);
	return ret;
}

/* Please note that device_id can be the ID of a camera or the PATH of a previously-recorded .rrf file */
PyCameraDevice *PyCameraManager::create_camera_device(PyObject *device_id) {
	if (!PyBytes_Check(device_id) && !PyUnicode_Check(device_id)) {
		PyErr_SetString(PyExc_TypeError, "Expecting a string value (unicode or bytes_array)");
		return NULL;
	}
	if (PyBytes_Check(device_id))
		camera = manager->createCamera(PyBytes_AsString(device_id));
	else
		camera = manager->createCamera(PyUnicode_AsUTF8(device_id));
	if (camera == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "Failed to create camera object");
		return NULL;
	}
	else {
		PyCameraDevice *pycamera = new PyCameraDevice(camera);
		logging("Created camera handle: %s\n", pycamera->get_camera_name());
		//return PyBytes_FromString(cameraname.c_str());
		return pycamera;
	}
}


////////////////////////////////////////////////////////////////////////////////


// Callbacks

template<typename... Args>
void call_python_callback(PyObject *callback, Args... vars) {
	PyObject *args = PyTuple_Pack(sizeof...(vars), vars...);
	PyObject *ret = PyObject_Call(callback, args, NULL);
	Py_DECREF(ret);		// ToDo Check why this is needed
	Py_DECREF(args);	// ToDo Check why this is needed
}

template<typename T>
PyObject *convert_buffer_to_numpy_array(royale::Vector<T> &data, royale::Vector<npy_intp> &dims, const NPY_TYPES type) {
	if (PyArray_API == NULL) {
		import_array();
	}

	return PyArray_SimpleNewFromData(2, dims.data(), type, data.data());
}

template<typename T>
PyObject *convert_buffer_to_numpy_array(royale::Vector<T> *data, royale::Vector<npy_intp> &dims, const NPY_TYPES type) {
	if (PyArray_API == NULL) {
		import_array();
	}

	return PyArray_SimpleNewFromData(2, dims.data(), type, data->data());
}


PyDataListener::PyDataListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyDataListener::~PyDataListener() {
	Py_XDECREF(callback);
}

void PyDataListener::onNewData(const royale::DepthData *data) {
	PyObject *timestamp = PyLong_FromUnsignedLongLong(data->timeStamp.count());
	PyObject *stream_id = PyLong_FromSize_t(data->streamId);
	PyObject *width = PyLong_FromSize_t(data->width);
	PyObject *height = PyLong_FromSize_t(data->height);

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		// npy_intp dims[2] = { data->height, data->width };
		royale::Vector<npy_intp> dims({ data->height, data->width });
		royale::Vector<npy_intp> point_dims({ data->height, data->width, 4 });

		//royale::Vector<float> v_x = royale::Vector<float>(data->height * data->width);
		//royale::Vector<float> v_y = royale::Vector<float>(data->height * data->width);
		//royale::Vector<float> v_z = royale::Vector<float>(data->height * data->width);
		//royale::Vector<float> v_noise = royale::Vector<float>(data->height * data->width);
		royale::Vector<uint16_t> v_gray = royale::Vector<uint16_t>(data->height * data->width);
		royale::Vector<uint8_t> v_depth_confidence = royale::Vector<uint8_t>(data->height * data->width);
		royale::Vector<uint32_t> v_exposure_times = royale::Vector<uint32_t>(data->height * data->width);
		royale::Vector<float> v_points = royale::Vector<float>(data->height * data->width * 4);

		for (size_t i = 0; i < (size_t)(data->height * data->width); ++i) {
			//(*v_x)[i] = data->points[i].x;
			//(*v_y)[i] = data->points[i].y;
			//(*v_z)[i] = data->points[i].z;
			//(*v_noise)[i] = data->points[i].noise;
			v_points[i * 4 + 0] = data->points[i].x;
			v_points[i * 4 + 1] = data->points[i].y;
			v_points[i * 4 + 2] = data->points[i].z;
			v_points[i * 4 + 3] = data->points[i].noise;
			v_gray[i] = data->points[i].grayValue;
			v_depth_confidence[i] = data->points[i].depthConfidence;
		}

		//PyObject *x = convert_buffer_to_numpy_array(v_x, dims, NPY_TYPES::NPY_FLOAT);
		//PyObject *y = convert_buffer_to_numpy_array(v_y, dims, NPY_TYPES::NPY_FLOAT);
		//PyObject *z = convert_buffer_to_numpy_array(v_z, dims, NPY_TYPES::NPY_FLOAT);
		//PyObject *noise = convert_buffer_to_numpy_array(v_noise, dims, NPY_TYPES::NPY_FLOAT);
		PyObject *points= convert_buffer_to_numpy_array(v_points, point_dims, NPY_TYPES::NPY_FLOAT);
		PyObject *gray = convert_buffer_to_numpy_array(v_gray, dims, NPY_TYPES::NPY_USHORT);
		PyObject *depth_confidence = convert_buffer_to_numpy_array(v_depth_confidence, dims, NPY_TYPES::NPY_UBYTE);

		//call_python_callback(callback, x, y, z, noise, gray, depth_confidence);
		call_python_callback(callback, timestamp, stream_id, width, height, points, gray, depth_confidence);
		PyGILState_Release(lock);
	}
}


PyDepthImageListener::PyDepthImageListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyDepthImageListener::~PyDepthImageListener() {
	Py_XDECREF(callback);
}

void PyDepthImageListener::onNewData(const royale::DepthImage *data) {
	PyObject *timestamp = PyLong_FromUnsignedLongLong(data->timestamp);
	PyObject *stream_id = PyLong_FromSize_t(data->streamId);
	PyObject *width = PyLong_FromSize_t(data->width);
	PyObject *height = PyLong_FromSize_t(data->height);

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		royale::Vector<npy_intp> dims({ data->height, data->width });

		royale::Vector<uint16_t> v_depth = royale::Vector<uint16_t>(data->height * data->width);
		royale::Vector<uint8_t> v_depth_confidence = royale::Vector<uint8_t>(data->height * data->width);

		for (size_t i = 0; i < (size_t)(data->height * data->width); ++i) {
			v_depth[i] = data->data[i] & 0x1FFF;
			v_depth_confidence[i] = data->data[i] & 0xE000;
		}

		PyObject *depth = convert_buffer_to_numpy_array(v_depth, dims, NPY_TYPES::NPY_USHORT);
		PyObject *depth_confidence = convert_buffer_to_numpy_array(v_depth_confidence, dims, NPY_TYPES::NPY_UBYTE);

		call_python_callback(callback, timestamp, stream_id, width, height, depth, depth_confidence);
		PyGILState_Release(lock);
	}
}


PyIRImageListener::PyIRImageListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyIRImageListener::~PyIRImageListener() {
	Py_XDECREF(callback);
}

void PyIRImageListener::onNewData(const royale::IRImage *data) {
	PyObject *timestamp = PyLong_FromUnsignedLongLong(data->timestamp);
	PyObject *stream_id = PyLong_FromSize_t(data->streamId);
	PyObject *width = PyLong_FromSize_t(data->width);
	PyObject *height = PyLong_FromSize_t(data->height);

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		royale::Vector<npy_intp> dims({ data->height, data->width });
		
		royale::Vector<uint8_t> v_ir_image = royale::Vector<uint8_t>(data->height * data->width);
		v_ir_image = data->data;

		PyObject *ir_image = convert_buffer_to_numpy_array(v_ir_image, dims, NPY_TYPES::NPY_UBYTE);

		call_python_callback(callback, timestamp, stream_id, width, height, ir_image);
		PyGILState_Release(lock);
	}
}


PySparsePointCloudListener::PySparsePointCloudListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PySparsePointCloudListener::~PySparsePointCloudListener() {
	Py_XDECREF(callback);
}

void PySparsePointCloudListener::onNewData(const royale::SparsePointCloud *data) {
	PyObject *timestamp = PyLong_FromUnsignedLongLong(data->timestamp);
	PyObject *stream_id = PyLong_FromSize_t(data->streamId);
	PyObject *num_points = PyLong_FromSize_t(data->numPoints);

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		royale::Vector<npy_intp> dims({ (int)data->numPoints, 4 });
		
		royale::Vector<float> v_sparse_point_cloud = royale::Vector<float>(4 * data->numPoints);
		for (size_t i = 0; i < data->numPoints; ++i) {
			v_sparse_point_cloud[i * 4 + 0] = data->xyzcPoints[i * 4 + 0];
			v_sparse_point_cloud[i * 4 + 1] = data->xyzcPoints[i * 4 + 1];
			v_sparse_point_cloud[i * 4 + 2] = data->xyzcPoints[i * 4 + 2];
			v_sparse_point_cloud[i * 4 + 3] = data->xyzcPoints[i * 4 + 3];
		}
		PyObject *sparse_point_cloud = convert_buffer_to_numpy_array(v_sparse_point_cloud, dims, NPY_TYPES::NPY_FLOAT);
		
		call_python_callback(callback, timestamp, stream_id, num_points, sparse_point_cloud);
		PyGILState_Release(lock);
	}
}


PyEventListener::PyEventListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyEventListener::~PyEventListener() {
	Py_XDECREF(callback);
}

void PyEventListener::onEvent(std::unique_ptr<royale::IEvent> &&event) {
	PyObject *event_severity = PyLong_FromLong((int)event->severity());
	PyObject *description = PyBytes_FromString(event->describe().data());
	PyObject *type = PyLong_FromLong((int)event->type());

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		call_python_callback(callback, event_severity, description, type);
		PyGILState_Release(lock);
	}
}


PyRecordStopListener::PyRecordStopListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyRecordStopListener::~PyRecordStopListener() {
	Py_XDECREF(callback);
}

void PyRecordStopListener::onRecordingStopped(const uint32_t numFrames) {
	PyObject *num_frames = PyLong_FromSize_t(numFrames);
	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		call_python_callback(callback, num_frames);
		PyGILState_Release(lock);
	}
}


PyExposureListener::PyExposureListener(PyObject *callback) {
	this->callback = callback;
	Py_XINCREF(this->callback);
}

PyExposureListener::~PyExposureListener() {
	Py_XDECREF(callback);
}

void PyExposureListener::onNewExposure(const uint32_t exposureTime, const royale::StreamId streamId) {
	PyObject *exposure_time = PyLong_FromSize_t(exposureTime);
	PyObject *stream_id = PyLong_FromSize_t(streamId);

	if (callback != NULL) {
		PyGILState_STATE lock = PyGILState_Ensure();

		call_python_callback(callback, exposure_time, stream_id);
		PyGILState_Release(lock);
	}
}



////////////////////////////////////////////////////////////////////////////////


/*
// *TODO
// Level > 1 APIs
*/



PyCameraDevice::PyCameraDevice(std::unique_ptr<royale::ICameraDevice> &camera) : camera(std::move(camera)) {
	royale::CameraStatus status;
	status = this->camera->getCameraName(this->name);
	if (status != royale::CameraStatus::SUCCESS) {
		set_error_message(status, "Failed to get camera name during the object initialization", PyExc_RuntimeError);
	}
}

PyCameraDevice::~PyCameraDevice() {
	camera.release();
}

PyObject *PyCameraDevice::initialize() {
	logging("Initializing camera device: %s.\n", this->name.c_str());
	royale::CameraStatus status = camera->initialize();
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to initialize camera device", PyExc_RuntimeError);
		return NULL;
	}
}


PyObject *PyCameraDevice::get_id() const {
	royale::String str;
	royale::CameraStatus status = camera->getId(str);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyBytes_FromString(str.data());
	}
	else {
		set_error_message(status, "Failed to get the camera id", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_camera_name() const {
	royale::String cameraName;

	royale::CameraStatus status = camera->getCameraName(cameraName);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyBytes_FromString(cameraName.data());
	}
	else {
		set_error_message(status, "Failed to get the camera name", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_camera_info() const {
	royale::Vector<royale::Pair<royale::String, royale::String>> cameraInfo;

	royale::CameraStatus status = camera->getCameraInfo(cameraInfo);
	if (status == royale::CameraStatus::SUCCESS) {
		return to_python_dict(cameraInfo);
	}
	else {
		set_error_message(status, "Failed to get the camera info", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_use_cases() const {
	royale::Vector<royale::String> useCases;

	royale::CameraStatus status = camera->getUseCases(useCases);
	if (status == royale::CameraStatus::SUCCESS) {
		return to_python_list(useCases);
	}
	else {
		set_error_message(status, "Failed to get the camera use cases", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::set_use_case(PyObject *name) {
	if (!PyBytes_Check(name) && !PyUnicode_Check(name)) {
		PyErr_SetString(PyExc_TypeError, "Expecting a string value (unicode or bytes_array)");
		return NULL;
	}

	royale::CameraStatus status;
	if (PyBytes_Check(name))
		status = camera->setUseCase(PyBytes_AsString(name));
	else
		status = camera->setUseCase(PyUnicode_AsUTF8(name));
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to set camera use case", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_number_of_streams(PyObject *name) const {
	if (!PyBytes_Check(name) && !PyUnicode_Check(name)) {
		PyErr_SetString(PyExc_TypeError, "Expecting a string value (unicode or bytes_array)");
		return NULL;
	}

	uint32_t nrStreams = 0;

	royale::CameraStatus status;
	if (PyBytes_Check(name))
		status = camera->getNumberOfStreams(PyBytes_AsString(name), nrStreams);
	else
		status = camera->getNumberOfStreams(PyUnicode_AsUTF8(name), nrStreams);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyLong_FromSize_t(nrStreams);
	}
	else {
		set_error_message(status, "Failed to set camera use case", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_streams() const {
	royale::Vector<royale::StreamId> streams;
	royale::Vector<double> streams_double;

	royale::CameraStatus status = camera->getStreams(streams);
	if (status == royale::CameraStatus::SUCCESS) {
		for (size_t i = 0; i < streams.size(); ++i) {
			streams_double.push_back(streams[i]);
		}
		return to_python_list(streams_double);
	}
	else {
		set_error_message(status, "Failed to set camera use case", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_current_use_case() const {
	royale::String useCase;

	royale::CameraStatus status = camera->getCurrentUseCase(useCase);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyBytes_FromString(useCase.data());
	}
	else {
		set_error_message(status, "Failed to get the current use case", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::set_exposure_time(PyObject *exposure_time, PyObject *stream_id /*= NULL*/) {
	if (stream_id == NULL)
		stream_id = PyLong_FromSize_t(0);
	if (!PyLong_Check(exposure_time)) {
		PyErr_SetString(PyExc_TypeError, "exposure_time must be integer or long");
		return NULL;
	}
	if (!PyLong_Check(stream_id)) {
		PyErr_SetString(PyExc_TypeError, "stream_id must be integer or long (default=0)");
		return NULL;
	}
	
	royale::CameraStatus status = camera->setExposureTime(PyLong_AsLong(exposure_time), PyLong_AsSize_t(stream_id));
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to set the exposure time", PyExc_RuntimeError);
		return NULL;
	}
}

// ToDo Consider to write a more user-friendly interface
PyObject *PyCameraDevice::set_exposure_mode(PyObject *exposure_mode, PyObject *stream_id /*= NULL*/) {
	if (stream_id == NULL)
		stream_id = PyLong_FromSize_t(0);
	if (!PyLong_Check(exposure_mode) || !(PyLong_AsUnsignedLong(exposure_mode) == EXPOSURE_MODE_MANUAL || PyLong_AsUnsignedLong(exposure_mode) == EXPOSURE_MODE_AUTO)) {
		PyErr_SetString(PyExc_TypeError, "exposure_mode must be 0 (MANUAL) or 1 (AUTOMATIC)");
		return NULL;
	}
	if (!PyLong_Check(stream_id)) {
		PyErr_SetString(PyExc_TypeError, "stream_id must be integer or long (default=0)");
		return NULL;
	}

	royale::CameraStatus status = camera->setExposureMode(royale::ExposureMode(PyLong_AsLong(exposure_mode)), PyLong_AsSize_t(stream_id));
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to set the exposure mode", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_exposure_mode(PyObject *stream_id /*= NULL*/) const {
	if (stream_id == NULL)
		stream_id = PyLong_FromLong(0);
	if (!PyLong_Check(stream_id)) {
		PyErr_SetString(PyExc_TypeError, "stream_id must be integer or long (default=0)");
		return NULL;
	}

	royale::ExposureMode exposureMode;

	royale::CameraStatus status = camera->getExposureMode(exposureMode, PyLong_AsSize_t(stream_id));
	if (status == royale::CameraStatus::SUCCESS) {
		if (exposureMode == royale::ExposureMode::MANUAL)
			return PyLong_FromLong(EXPOSURE_MODE_MANUAL);
		else
			return PyLong_FromLong(EXPOSURE_MODE_AUTO);
	}
	else {
		set_error_message(status, "Failed to get the exposure mode", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_exposure_limits(PyObject *stream_id /*= NULL*/) const {
	if (stream_id == NULL)
		stream_id = PyLong_FromSize_t(0);
	if (!PyLong_Check(stream_id)) {
		PyErr_SetString(PyExc_TypeError, "stream_id must be integer or long (default=0)");
		return NULL;
	}

	royale::Pair<uint32_t, uint32_t> exposureLimits;

	royale::CameraStatus status = camera->getExposureLimits(exposureLimits, PyLong_AsSize_t(stream_id));
	if (status == royale::CameraStatus::SUCCESS) {
		return PyTuple_Pack(2, exposureLimits.first, exposureLimits.second);
	}
	else {
		set_error_message(status, "Failed to get the exposure limits", PyExc_RuntimeError);
		return NULL;
	}
}


PyObject *PyCameraDevice::register_data_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	data_listener = new PyDataListener(callback);
	
	royale::CameraStatus status = camera->registerDataListener(data_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the data listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_data_listener() {
	royale::CameraStatus status = camera->unregisterDataListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete data_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the data listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::register_depth_image_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	depth_data_listener = new PyDepthImageListener(callback);

	royale::CameraStatus status = camera->registerDepthImageListener(depth_data_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the depth image listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_depth_image_listener() {
	royale::CameraStatus status = camera->unregisterDepthImageListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete depth_data_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the depth image listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::register_sparse_point_cloud_listener(PyObject *callback) {
	// royale::ISparsePointCloudListener *listener
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	sparse_point_cloud_listener = new PySparsePointCloudListener(callback);

	royale::CameraStatus status = camera->registerSparsePointCloudListener(sparse_point_cloud_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the sparse point cloud listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_sparse_point_cloud_listener() {
	royale::CameraStatus status = camera->unregisterSparsePointCloudListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete sparse_point_cloud_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the sparse point cloud listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::register_ir_image_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	ir_image_listener = new PyIRImageListener(callback);

	royale::CameraStatus status = camera->registerIRImageListener(ir_image_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the IR image listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_ir_image_listener() {
	royale::CameraStatus status = camera->unregisterIRImageListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete ir_image_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the IR image listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::register_event_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	event_listener = new PyEventListener(callback);

	royale::CameraStatus status = camera->registerEventListener(event_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the event listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_event_listener() {
	royale::CameraStatus status = camera->unregisterEventListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete event_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the event listener", PyExc_RuntimeError);
		return NULL;
	}
}


PyObject *PyCameraDevice::start_capture() {
	royale::CameraStatus status = camera->startCapture();
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to start the capture", PyExc_RuntimeError);
		Py_RETURN_NONE;
	}
}

PyObject *PyCameraDevice::stop_capture() {
	royale::CameraStatus status = camera->stopCapture();
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to stop the capture", PyExc_RuntimeError);
		Py_RETURN_NONE;
	}
}

PyObject *PyCameraDevice::get_max_sensor_width() const {
	uint16_t maxSensorWidth;

	royale::CameraStatus status = camera->getMaxSensorWidth(maxSensorWidth);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyLong_FromSize_t(maxSensorWidth);
	}
	else {
		set_error_message(status, "Failed to get the max sensor width", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_max_sensor_height() const {
	uint16_t maxSensorHeight;

	royale::CameraStatus status = camera->getMaxSensorWidth(maxSensorHeight);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyLong_FromSize_t(maxSensorHeight);
	}
	else {
		set_error_message(status, "Failed to get the max sensor height", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_lens_parameters() const {
	royale::LensParameters param;

	royale::CameraStatus status = camera->getLensParameters(param);
	if (status == royale::CameraStatus::SUCCESS) {
		royale::Vector<royale::Pair<royale::String, double>> pairs;
		pairs.push_back(royale::Pair<royale::String, double>("principalPoint_cx", param.principalPoint.first));
		pairs.push_back(royale::Pair<royale::String, double>("principalPoint_cy", param.principalPoint.second));
		pairs.push_back(royale::Pair<royale::String, double>("focalLength_fx", param.focalLength.first));
		pairs.push_back(royale::Pair<royale::String, double>("focalLength_fy", param.focalLength.second));
		pairs.push_back(royale::Pair<royale::String, double>("distortionTangential_p1", param.distortionTangential.first));
		pairs.push_back(royale::Pair<royale::String, double>("distortionTangential_p2", param.distortionTangential.second));
		pairs.push_back(royale::Pair<royale::String, double>("distortionRadial_k1", param.distortionRadial[0]));
		pairs.push_back(royale::Pair<royale::String, double>("distortionRadial_k2", param.distortionRadial[1]));
		pairs.push_back(royale::Pair<royale::String, double>("distortionRadial_k3", param.distortionRadial[2]));

		return to_python_dict(pairs);
	}
	else {
		set_error_message(status, "Failed to get the lens parameters", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::is_connected() const {
	bool connected;

	royale::CameraStatus status = camera->isConnected(connected);
	if (status == royale::CameraStatus::SUCCESS) {
		//return connected ? Py_True : Py_False;
		return PyBool_FromLong(connected);
	}
	else {
		set_error_message(status, "Failed to check if camera is connected", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::is_calibrated() const {
	bool calibrated;

	royale::CameraStatus status = camera->isCalibrated(calibrated);
	if (status == royale::CameraStatus::SUCCESS) {
		//return calibrated ? Py_True : Py_False;
		return PyBool_FromLong(calibrated);
	}
	else {
		set_error_message(status, "Failed to check if camera is calibrated", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::is_capturing() const {
	bool capturing;

	royale::CameraStatus status = camera->isCapturing(capturing);
	if (status == royale::CameraStatus::SUCCESS) {
		//return capturing ? Py_True : Py_False;
		return PyBool_FromLong(capturing);
	}
	else {
		set_error_message(status, "Failed to check if camera is capturing", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_access_level() const {
	royale::CameraAccessLevel accessLevel;

	royale::CameraStatus status = camera->getAccessLevel(accessLevel);
	if (status == royale::CameraStatus::SUCCESS) {
		switch (accessLevel) {
		case royale::CameraAccessLevel::L1:
			return PyLong_FromLong(CAMERA_ACCESS_LEVEL_L1);
			break;
		case royale::CameraAccessLevel::L2:
			return PyLong_FromLong(CAMERA_ACCESS_LEVEL_L2);
			break;
		case royale::CameraAccessLevel::L3:
			return PyLong_FromLong(CAMERA_ACCESS_LEVEL_L3);
			break;
		case royale::CameraAccessLevel::L4:
			return PyLong_FromLong(CAMERA_ACCESS_LEVEL_L4);
			break;
		default:
			set_error_message(status, "Returned an undefined camera access level", PyExc_ValueError);
			return NULL;
		}
	}
	else {
		set_error_message(status, "Failed to get the camera access level", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::start_recording(PyObject *file_name, PyObject *number_of_frames /*= NULL*/, PyObject *frame_skip /*= NULL*/, PyObject *ms_skip /*= NULL*/) {
	if (number_of_frames == NULL)
		number_of_frames = PyLong_FromSize_t(0);
	if (frame_skip == NULL)
		frame_skip = PyLong_FromSize_t(0);
	if (ms_skip == NULL)
		ms_skip = PyLong_FromSize_t(0);
	if (!PyBytes_Check(file_name) && !PyUnicode_Check(file_name)) {
		PyErr_SetString(PyExc_TypeError, "file_name must be a string (unicode or bytes_array)");
		return NULL;
	}
	if (!PyLong_Check(number_of_frames)) {
		PyErr_SetString(PyExc_TypeError, "number_of_frames must be integer or long (default=0)");
		return NULL;
	}
	if (!PyLong_Check(frame_skip)) {
		PyErr_SetString(PyExc_TypeError, "frame_skip must be integer or long (default=0)");
		return NULL;
	}
	if (!PyLong_Check(ms_skip)) {
		PyErr_SetString(PyExc_TypeError, "ms_skip must be integer or long (default=0)");
		return NULL;
	}

	royale::CameraStatus status;
	if (PyBytes_Check(file_name))
		status = camera->startRecording(PyBytes_AsString(file_name), PyLong_AsSize_t(number_of_frames), PyLong_AsSize_t(frame_skip), PyLong_AsSize_t(ms_skip));
	else
		status = camera->startRecording(PyUnicode_AsUTF8(file_name), PyLong_AsSize_t(number_of_frames), PyLong_AsSize_t(frame_skip), PyLong_AsSize_t(ms_skip));
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to start recording", PyExc_RuntimeError);
		Py_RETURN_NONE;
	}
}

PyObject *PyCameraDevice::stop_recording() {
	royale::CameraStatus status = camera->stopRecording();
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to stop recording", PyExc_RuntimeError);
		Py_RETURN_NONE;
	}
}


PyObject *PyCameraDevice::register_record_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	record_listener = new PyRecordStopListener(callback);

	royale::CameraStatus status = camera->registerExposureListener(exposure_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the exposure listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_record_listener() {
	royale::CameraStatus status = camera->unregisterRecordListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete record_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the IR image listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::register_exposure_listener(PyObject *callback) {
	if (!PyCallable_Check(callback)) {
		PyErr_SetString(PyExc_TypeError, "callback must be a callable object.");
		return NULL;
	}

	exposure_listener = new PyExposureListener(callback);

	royale::CameraStatus status = camera->registerExposureListener(exposure_listener);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to register the exposure listener", PyExc_RuntimeError);
		return NULL;
	}
}

PyObject *PyCameraDevice::unregister_exposure_listener() {
	royale::CameraStatus status = camera->unregisterExposureListener();
	if (status == royale::CameraStatus::SUCCESS) {
		delete exposure_listener;
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to unregister the exposure listener", PyExc_RuntimeError);
		return NULL;
	}
}


PyObject *PyCameraDevice::set_frame_rate(PyObject *frame_rate) {
	if (!PyLong_Check(frame_rate)) {
		PyErr_SetString(PyExc_TypeError, "frame_rate must be integer or long");
		return NULL;
	}

	royale::CameraStatus status = camera->setFrameRate(PyLong_AsSize_t(frame_rate));
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to set the frame rate", PyExc_RuntimeError);
		Py_RETURN_NONE;
	}
}

PyObject *PyCameraDevice::get_frame_rate() const {
	uint16_t frameRate;

	royale::CameraStatus status = camera->getFrameRate(frameRate);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyLong_FromSize_t(frameRate);
	}
	else {
		set_error_message(status, "Failed to get the frame rate", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::get_max_frame_rate() const {
	uint16_t maxFrameRate;

	royale::CameraStatus status = camera->getMaxFrameRate(maxFrameRate);
	if (status == royale::CameraStatus::SUCCESS) {
		return PyLong_FromSize_t(maxFrameRate);
	}
	else {
		set_error_message(status, "Failed to get the max frame rate", PyExc_ValueError);
		return NULL;
	}
}

PyObject *PyCameraDevice::set_external_trigger(PyObject *use_external_trigger) {
	if (!PyBool_Check(use_external_trigger)) {
		PyErr_SetString(PyExc_TypeError, "use_external_trigger must be True or False");
		return NULL;
	}
	
	// ToDo check if PyBool_FromLong(1) == Py_True
	royale::CameraStatus status = camera->setExternalTrigger(use_external_trigger == Py_True);
	if (status == royale::CameraStatus::SUCCESS) {
		Py_RETURN_NONE;
	}
	else {
		set_error_message(status, "Failed to set the external trigger", PyExc_ValueError);
		return NULL;
	}
}

