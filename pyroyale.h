#ifndef _PYROYALE_H
#define _PYROYALE_H

#define SWIG_FILE_WITH_INIT
//#define ROYALE_C_API_VERSION 31000
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

#include <royale.hpp>
#include <royale/IEvent.hpp>

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
PyObject *to_list_of_double(royale::Vector<double> &elems);

/// Set Python error message from royale error status
void set_error_message(const royale::CameraStatus status, const char *message, PyObject *error_type);

/// Logging function
void logging(const char * format, ...);
*/


class PyCameraManager;
class PyCameraDevice;
class PyDataListener;
class PyDepthImageListener;
class PyIRImageListener;
class PySparsePointCloudListener;
class PyEventListener;


class PyCameraManager {
private:
	royale::CameraManager *manager;
	std::unique_ptr<royale::ICameraDevice> camera;
public:
	PyCameraManager();
	~PyCameraManager();
	PyObject *initialize();
	PyObject *get_connected_cameras() const;
	PyCameraDevice *create_camera_device(PyObject *device_id);
};


////////////////////////////////////////////////////////////////////////////////


// Callbacks

class PyDataListener : public royale::IDepthDataListener {
private:
	PyObject *callback;
public:
	PyDataListener(PyObject *callback);
	~PyDataListener();

	void onNewData(const royale::DepthData *data);
};

class PyDepthImageListener : public royale::IDepthImageListener {
private:
	PyObject *callback;
public:
	PyDepthImageListener(PyObject *callback);
	~PyDepthImageListener();

	void onNewData(const royale::DepthImage *data);
};

class PyIRImageListener : public royale::IIRImageListener {
private:
	PyObject *callback;
public:
	PyIRImageListener(PyObject *callback);
	~PyIRImageListener();

	void onNewData(const royale::IRImage *data);
};

class PySparsePointCloudListener : public royale::ISparsePointCloudListener {
private:
	PyObject *callback;
public:
	PySparsePointCloudListener(PyObject *callback);
	~PySparsePointCloudListener();

	void onNewData(const royale::SparsePointCloud *data);
};

class PyEventListener : public royale::IEventListener {
private:
	PyObject *callback;
public:
	PyEventListener(PyObject *callback);
	~PyEventListener();

	void onEvent(std::unique_ptr<royale::IEvent> &&event);
};

class PyRecordStopListener : public royale::IRecordStopListener {
private:
	PyObject *callback;
public:
	PyRecordStopListener(PyObject *callback);
	~PyRecordStopListener();

	void onRecordingStopped(const uint32_t numFrames);
};

class PyExposureListener : public royale::IExposureListener2 {
private:
	PyObject *callback;
public:
	PyExposureListener(PyObject *callback);
	~PyExposureListener();

	void onNewExposure(const uint32_t exposureTime, const royale::StreamId streamId);
};

/*
void call_python_callback(PyObject *callback, PyObject *x_array, PyObject *y_array, PyObject *z_array, PyObject *noise, PyObject *gray_array, PyObject *depth_confidence);

template<typename T>
PyObject *convert_buffer_to_numpy_array(const royale::Vector<T> &data, const royale::Vector<npy_intp> &dims, const NPY_TYPES type);
*/

////////////////////////////////////////////////////////////////////////////////


class PyCameraDevice {
	std::unique_ptr<royale::ICameraDevice> camera;
	royale::String name;
	PyDataListener *data_listener;
	PyDepthImageListener *depth_data_listener;
	PyIRImageListener *ir_image_listener;
	PySparsePointCloudListener *sparse_point_cloud_listener;
	PyEventListener *event_listener;
	PyRecordStopListener *record_listener;
	PyExposureListener *exposure_listener;
public:
	PyCameraDevice(std::unique_ptr<royale::ICameraDevice> &camera);
	~PyCameraDevice();
	PyObject *initialize();


	/*!
	*  LEVEL 1
	*  Get the ID of the camera device
	*/
	PyObject *get_id() const;

	/*!
	*  LEVEL 1
	*  Returns the associated camera name as a string which is defined in the CoreConfig
	*  of each module.
	*/
	PyObject *get_camera_name() const;

	/*!
	*  LEVEL 1
	* Retrieve further information for this specific camera.
	* The return value is a map, where the keys are depending on the used camera
	*/
	PyObject *get_camera_info() const;

	/*!
	*  LEVEL 1
	*  Returns all use cases which are supported by the connected module and valid for the
	*  current selected CallbackData information (e.g. Raw, Depth, ...)
	*/
	PyObject *get_use_cases() const;

	/*!
	*  LEVEL 1
	*  Sets the use case for the camera. If the use case is supported by the connected
	*  camera device SUCCESS will be returned. Changing the use case will also change the
	*  processing parameters that are used (e.g. auto exposure)!
	*
	*  NOTICE: This function must not be called in the data callback - the behavior is
	*  undefined.  Call it from a different thread instead.
	*
	* \param name identifies the use case by an case sensitive string
	*
	* \return SUCCESS if use case can be set
	*/
	PyObject *set_use_case(PyObject *name);

	/*!
	*  LEVEL 1
	*  Get the streams associated with the current use case.
	*/
	PyObject *get_streams() const;

	/*!
	*  LEVEL 1
	*  Retrieves the number of streams for a specified use case.
	*
	*  \param name use case name
	*  \param nrStreams number of streams for the specified use case
	*/
	PyObject *get_number_of_streams(PyObject *name) const;
	
	/*!
	*  LEVEL 1
	*  Gets the current use case as string
	*
	*  \param useCase current use case identified as string
	*/
	PyObject *get_current_use_case() const;

	/*!
	*  LEVEL 1
	*  Change the exposure time for the supported operated operation modes.
	*
	*  For mixed-mode use cases a valid streamId must be passed.
	*  For use cases having only one stream the default value of 0 (which is otherwise not a valid
	*  stream id) can be used to refer to that stream. This is for backward compatibility.
	*
	*  If MANUAL exposure mode of operation is chosen, the user is able to determine set
	*  exposure time manually within the boundaries of the exposure limits of the specific
	*  operation mode.
	*
	*  On success the corresponding status message is returned.
	*  In any other mode of operation the method will return EXPOSURE_MODE_INVALID to indicate
	*  non-compliance with the selected exposure mode.
	*  If the camera is used in the playback configuration a LOGIC_ERROR is returned instead.
	*
	*  WARNING : If this function is used on Level 3 it will ignore the limits given by the use case.
	*
	*  \param exposureTime exposure time in microseconds
	*  \param streamId which stream to change exposure for
	*/
	PyObject *set_exposure_time(PyObject *exposure_time, PyObject *stream_id = NULL);

	/*!
	*  LEVEL 1
	*  Change the exposure mode for the supported operated operation modes.
	*
	*  For mixed-mode use cases a valid streamId must be passed.
	*  For use cases having only one stream the default value of 0 (which is otherwise not a valid
	*  stream id) can be used to refer to that stream. This is for backward compatibility.
	*
	*  If MANUAL exposure mode of operation is chosen, the user is able to determine set
	*  exposure time manually within the boundaries of the exposure limits of the specific
	*  operation mode.
	*
	*  In AUTOMATIC mode the optimum exposure settings are determined the system itself.
	*
	*  The default value is MANUAL.
	*
	*  \param exposureMode mode of operation to determine the exposure time
	*  \param streamId which stream to change exposure mode for
	*/
	PyObject *set_exposure_mode(PyObject *exposure_mode, PyObject *stream_id = NULL);

	/*!
	*  LEVEL 1
	*  Retrieves the current mode of operation for acquisition of the exposure time.
	*
	*  For mixed-mode usecases a valid streamId must be passed.
	*  For usecases having only one stream the default value of 0 (which is otherwise not a valid
	*  stream id) can be used to refer to that stream. This is for backward compatibility.
	*
	*  \param exposureMode contains current exposure mode on successful return
	*  \param streamId stream for which the exposure mode should be returned
	*/
	PyObject *get_exposure_mode(PyObject *stream_id = NULL) const;

	/*!
	*  LEVEL 1
	*  Retrieves the minimum and maximum allowed exposure limits of the specified operation
	*  mode.  Can be used to retrieve the allowed operational range for a manual definition of
	*  the exposure time.
	*
	*  For mixed-mode usecases a valid streamId must be passed.
	*  For usecases having only one stream the default value of 0 (which is otherwise not a valid
	*  stream id) can be used to refer to that stream. This is for backward compatibility.
	*
	*  \param exposureLimits contains the limits on successful return
	*  \param streamId stream for which the exposure limits should be returned
	*/
	PyObject *get_exposure_limits(PyObject *stream_id = NULL) const;

	/**
	*  LEVEL 1
	*  Once registering the data listener, 3D point cloud data is sent via the callback
	*  function.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_data_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters the data depth listener
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_data_listener();

	/**
	*  LEVEL 1
	*  Once registering the data listener, Android depth image data is sent via the
	*  callback function.
	*
	*  Consider using registerDataListener and an IDepthDataListener instead of this listener.
	*  This callback provides only an array of depth and confidence values.  The mapping of
	*  pixels to the scene is similar to the pixels of a two-dimensional camera, and it is
	*  unlikely to be a rectilinear projection (although this depends on the exact camera).
	*
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_depth_image_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters the depth image listener
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_depth_image_listener();

	/**
	*  LEVEL 1
	*  Once registering the data listener, Android point cloud data is sent via the
	*  callback function.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_sparse_point_cloud_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters the sparse point cloud listener
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_sparse_point_cloud_listener();

	/**
	*  LEVEL 1
	*  Once registering the data listener, IR image data is sent via the callback function.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_ir_image_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters the IR image listener
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_ir_image_listener();

	/**
	*  LEVEL 1
	*  Register listener for event notifications.
	*  The callback will be invoked asynchronously.
	*  Events include things like illumination unit overtemperature.
	*/
	PyObject *register_event_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters listener for event notifications.
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_event_listener();

	/**
	*  LEVEL 1
	*  Starts the video capture mode (free-running), based on the specified operation mode.
	*  A listener needs to be registered in order to retrieve the data stream. Either raw data
	*  or processed data can be consumed. If no data listener is registered an error will be
	*  returned and capturing is not started.
	*/
	PyObject *start_capture();

	/**
	*  LEVEL 1
	*  Stops the video capturing mode.
	*  All buffers should be released again by the data listener.
	*/
	PyObject *stop_capture();

	/**
	*  LEVEL 1
	*  Returns the maximal width supported by the camera device.
	*/
	PyObject *get_max_sensor_width() const;

	/**
	*  LEVEL 1
	*  Returns the maximal height supported by the camera device.
	*/
	PyObject *get_max_sensor_height() const;

	/**
	*  LEVEL 1
	*  Gets the intrinsics of the camera module which are stored in the calibration file
	*
	*  \param param LensParameters is storing all the relevant information (c,f,p,k)
	*
	*  \return CameraStatus
	*/
	PyObject *get_lens_parameters() const;

	/**
	*  LEVEL 1
	*  Returns the information if a connection to the camera could be established
	*
	*  \param connected true if properly set up
	*/
	PyObject *is_connected() const;

	/**
	*  LEVEL 1
	*  Returns the information if the camera module is calibrated. Older camera modules
	*  can still be operated with royale, but calibration data may be incomplete.
	*
	*  \param calibrated true if the module contains proper calibration data
	*/
	PyObject *is_calibrated() const;

	/**
	*  LEVEL 1
	*  Returns the information if the camera is currently in capture mode
	*
	*  \param capturing true if camera is in capture mode
	*/
	PyObject *is_capturing() const;

	/*!
	*  LEVEL 1
	*  Returns the current camera device access level
	*/
	PyObject *get_access_level() const;

	/*!
	*  LEVEL 1
	*  Start recording the raw data stream into a file.
	*  The recording will capture the raw data coming from the imager.
	*  If frameSkip and msSkip are both zero every frame will be recorded.
	*  If both are non-zero the behavior is implementation-defined.
	*
	*  \param fileName full path of target filename (proposed suffix is .rrf)
	*  \param numberOfFrames indicate the maximal number of frames which should be captured
	*                        (stop will be called automatically). If zero (default) is set,
	*                        recording will happen till stopRecording is called.
	*  \param frameSkip indicate how many frames should be skipped after every recorded frame.
	*                   If zero (default) is set and msSkip is zero, every frame will be
	*                   recorded.
	*  \param msSkip indicate how many milliseconds should be skipped after every recorded
	*                frame. If zero (default) is set and frameSkip is zero, every frame will
	*                be recorded.
	*/
	PyObject *start_recording(PyObject *fileName, PyObject *numberOfFrames = NULL, PyObject *frameSkip = NULL, PyObject *msSkip = NULL);

	/*!
	*  LEVEL 1
	*  Stop recording the raw data stream into a file. After the recording is stopped
	*  the file is available on the file system.
	*/
	PyObject *stop_recording();

	/**
	*  LEVEL 1
	*  Once registering a record listener, the listener gets notified once recording
	*  has stopped after specified frames.
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_record_listener(PyObject *callback);

	/**
	*  LEVEL 1
	*  Unregisters the record listener.
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_record_listener();

	/*!
	*  LEVEL 1
	*  [deprecated]
	*  Once registering the exposure listener, new exposure values calculated by the
	*  processing are sent to the listener. As this listener doesn't support streams,
	*  only updates for the first stream will be sent.
	*
	*  Only one exposure listener is supported at a time, calling this will automatically
	*  unregister any previously registered IExposureListener or IExposureListener2.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	//PyObject *register_exposure_listener(royale::IExposureListener *listener);	// deprecated

	/*!
	*  LEVEL 1
	*  Once registering the exposure listener, new exposure values calculated by the
	*  processing are sent to the listener.
	*
	*  Only one exposure listener is supported at a time, calling this will automatically
	*  unregister any previously registered IExposureListener or IExposureListener2.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	PyObject *register_exposure_listener(PyObject *callback);

	/*!
	*  LEVEL 1
	*  Unregisters the exposure listener
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	PyObject *unregister_exposure_listener();

	/*!
	*  LEVEL 1
	*  Set the frame rate to a value. Upper bound is given by the use case.
	*  E.g. Usecase with 5 FPS, a maximum frame rate of 5 and a minimum of 1 can be set.
	*  Setting a frame rate of 0 is not allowed.
	*
	*  The framerate is specific for the current use case.
	*  This function is not supported for mixed-mode.
	*/
	PyObject *set_frame_rate(PyObject *frame_rate);

	/*!
	*  LEVEL 1
	*  Get the current frame rate which is set for the current use case.
	*  This function is not supported for mixed-mode.
	*/
	PyObject *get_frame_rate() const;

	/*!
	*  LEVEL 1
	*  Get the maximal frame rate which can be set for the current use case.
	*  This function is not supported for mixed-mode.
	*/
	PyObject *get_max_frame_rate() const;

	/*!
	*  LEVEL 1
	*  Enable or disable the external triggering.
	*  Some camera modules support an external trigger, they can capture images synchronized with another device.
	*  If the hardware you are using supports it, calling setExternalTrigger(true) will make the camera capture images in this way.
	*  The call to setExternalTrigger has to be done before initializing the device.
	*
	*  The external signal must not exceed the maximum FPS of the chosen UseCase, but lower frame rates are supported.
	*  If no external signal is received, the imager will not start delivering images.
	*
	*  For information if your camera module supports external triggering and how to use it please refer to
	*  the Getting Started Guide of your camera. If the module doesn't support triggering calling this function
	*  will return a LOGIC_ERROR.
	*
	*  Royale currently expects a trigger pulse, not a constant trigger signal. Using a constant
	*  trigger signal might lead to a wrong framerate!
	*/
	PyObject *set_external_trigger(PyObject *use_external_trigger);

	// -----------------------------------------------------------------------------------------
	// Level 2: Experienced users (Laser Class 1 guaranteed) - activation key required
	// -----------------------------------------------------------------------------------------

	/*!
	*  LEVEL 2
	*  Get the list of exposure groups supported by the currently set use case.
	*/
	//PyObject *get_exposure_groups(royale::Vector<royale::String> &exposureGroups) const;

	/*!
	*  LEVEL 2
	*  Change the exposure time for the supported operated operation modes. If MANUAL exposure mode of operation is chosen, the user
	*  is able to determine set exposure time manually within the boundaries of the exposure limits of the specific operation mode.
	*  On success the corresponding status message is returned.
	*  In any other mode of operation the method will return EXPOSURE_MODE_INVALID to indicate incompliance with the
	*  selected exposure mode. If the camera is used in the playback configuration a LOGIC_ERROR is returned instead.
	*
	*  \param exposureGroup exposure group to be updated
	*  \param exposureTime exposure time in microseconds
	*/
	//PyObject *set_exposure_time(const royale::String &exposureGroup, uint32_t exposureTime);

	/*!
	*  LEVEL 2
	*  Retrieves the minimum and maximum allowed exposure limits of the specified operation mode.
	*  Limits may vary between exposure groups.
	*  Can be used to retrieve the allowed operational range for a manual definition of the exposure time.
	*
	*  \param exposureGroup exposure group to be queried
	*  \param exposureLimits pair of (minimum, maximum) exposure time in microseconds
	*/
	//PyObject *get_exposure_limits(const royale::String &exposureGroup, royale::Pair<uint32_t, uint32_t> &exposureLimits) const;

	/*!
	*  LEVEL 2
	*  Change the exposure times for all sequences.
	*  As it is possible to reuse an exposure group for different sequences it can happen
	*  that the exposure group is updated multiple times!
	*  If the vector that is provided is too long the extraneous values will be discard.
	*  If the vector is too short an error will be returned.
	*
	*  WARNING : If this function is used on Level 3 it will ignore the limits given by the use case.
	*
	*  \param exposureTimes vector with exposure times in microseconds
	*  \param streamId which stream to change exposure times for
	*/
	//PyObject *set_exposure_times(const royale::Vector<uint32_t> &exposureTimes, royale::StreamId streamId = 0);

	/*!
	*  LEVEL 2
	*  Change the exposure times for all exposure groups.
	*  The order of the exposure times is aligned with the order of exposure groups received by getExposureGroups.
	*  If the vector that is provided is too long the extraneous values will be discard.
	*  If the vector is too short an error will be returned.
	*
	*  \param exposureTimes vector with exposure times in microseconds
	*/
	//PyObject *set_exposure_for_groups(const royale::Vector<uint32_t> &exposureTimes);

	/*!
	*  LEVEL 2
	*  Set/alter processing parameters in order to control the data output.
	*  A list of processing flags is available as an enumeration. The `Variant` data type
	*  can take float, int, or bool. Please make sure to set the proper `Variant` type
	*  for the enum.
	*/
	//PyObject *set_processing_parameters(const royale::ProcessingParameterVector &parameters, uint16_t streamId = 0);

	/*!
	*  LEVEL 2
	*  Retrieve the available processing parameters which are used for the calculation.
	*
	*  Some parameters may only be available on some devices (and may depend on both the
	*  processing implementation and the calibration data available from the device), therefore
	*  the length of the vector may be less than ProcessingFlag::NUM_FLAGS.
	*/
	//PyObject *get_processing_parameters(royale::ProcessingParameterVector &parameters, uint16_t streamId = 0);

	/**
	*  LEVEL 2
	*  After registering the extended data listener, extended data is sent via the callback
	*  function.  If depth data only is specified, this listener is not called. For this case,
	*  please use the standard depth data listener.
	*
	*  \param listener interface which needs to implement the callback method
	*/
	//PyObject *register_data_listener_extended(royale::IExtendedDataListener *listener);

	/**
	*  LEVEL 2
	*  Unregisters the data extended listener.
	*
	*  It's not necessary to unregister this listener (or any other listener) before deleting
	*  the ICameraDevice.
	*/
	//PyObject *unregister_data_listener_extended();

	/**
	*  LEVEL 2
	*  Set the callback output data type to one type only.
	*
	*  INFO: This method needs to be called before startCapture(). If is is called while
	*  the camera is in capture mode, it will only have effect after the next stop/start
	*  sequence.
	*/
	//PyObject *set_callback_data(royale::CallbackData cbData);

	/**
	*  LEVEL 2
	*  [deprecated]
	*  Set the callback output data type. Setting multiple types currently isn't supported.
	*
	*  INFO: This method needs to be called before start_capture(). If is is called while
	*  the camera is in capture mode, it will only have effect after the next stop/start
	*  sequence.
	*/
	//PyObject *set_callback_data(uint16_t cbData);

	/**
	*  LEVEL 2
	*  Loads a different calibration from a file. This calibration data will also be used
	*  by the processing!
	*
	*  \param filename name of the calibration file which should be loaded
	*
	*  \return CameraStatus
	*/
	//PyObject *set_calibration_data(const royale::String &filename);

	/**
	*  LEVEL 2
	*  Loads a different calibration from a given Vector. This calibration data will also be
	*  used by the processing!
	*
	*  \param data calibration data which should be used
	*
	*  \return CameraStatus
	*/
	//PyObject *set_calibration_data(const royale::Vector<uint8_t> &data);

	/**
	*  LEVEL 2
	*  Retrieves the current calibration data.
	*
	*  \param data Vector which will be filled with the calibration data
	*/
	//PyObject *get_calibration_data(royale::Vector<uint8_t> &data);

	/**
	*  LEVEL 2
	*  Tries to write the current calibration file into the internal flash of the device.
	*  If no flash is found RESOURCE_ERROR is returned. If there are errors during the flash
	*  process it will try to restore the original calibration.
	*
	*  This is not yet implemented for all cameras!
	*
	*  Some devices also store other data in the calibration data area, for example the product
	*  identifier.  This L2 method will only change the calibration data, and will preserve the
	*  other data; if an unsupported combination of existing data and new data is encountered
	*  it will return an error without writing to the storage.  Only the L3 methods can change
	*  or remove the additional data.
	*
	*  \return CameraStatus
	*/
	//PyObject *write_calibration_to_flash();

	// -----------------------------------------------------------------------------------------
	// Level 3: Advanced users (Laser Class 1 not (!) guaranteed) - activation key required
	// -----------------------------------------------------------------------------------------

	/*!
	*  LEVEL 3
	*  Writes an arbitrary vector of data on to the storage of the device.
	*  If no flash is found RESOURCE_ERROR is returned.
	*
	*  Where the data will be written to is implementation defined. After using this function,
	*  the eye safety of the device is not guaranteed, even after reopening the device with L1
	*  access.  This method may overwrite the product identifier, and potentially even firmware
	*  in the device.
	*
	*  \param data data that should be flashed
	*/
	//PyObject *write_data_to_flash(royale::Vector<uint8_t> &data);

	/*!
	*  LEVEL 3
	*  Writes an arbitrary file to the storage of the device.
	*  If no flash is found RESOURCE_ERROR is returned.
	*
	*  Where the data will be written to is implementation defined. After using this function,
	*  the eye safety of the device is not guaranteed, even after reopening the device with L1
	*  access.  This method may overwrite the product identifier, and potentially even firmware
	*  in the device.
	*
	*  \param filename name of the file that should be flashed
	*/
	//PyObject *write_data_to_flash(const royale::String &filename);

	/*!
	*  LEVEL 3
	*  Change the dutycycle of a certain sequence. If the dutycycle is not supported,
	*  an error will be returned. The dutycycle can also be altered during capture
	*  mode.
	*
	*  \param dutyCycle dutyCycle in percent (0, 100)
	*  \param index index of the sequence to change
	*/
	//PyObject *set_duty_cycle(double dutyCycle, uint16_t index);

	/**
	* LEVEL 3
	* For each element of the vector a single register write is issued for the connected
	* imager.  Please be aware that any writes that will change crucial parts (starting the
	* imager, stopping the imager, changing the ROI, ...) will not be reflected internally by
	* Royale and might crash the program!
	*
	* If this function is used on Level 4 (empty imager), please be aware that Royale will not
	* start/stop the imager!
	*
	* USE AT YOUR OWN RISK!!!
	*
	* \param   registers   Contains elements of possibly not-unique (String, uint64_t) duplets.
	*                      The String component can consist of:
	*                      a) a base-10 decimal number in the range of [0, 65535]
	*                      b) a base-16 hexadecimal number preceded by a "0x" in the
	*                         range of [0, 65535]
	*/
	//PyObject *write_registers(const royale::Vector<royale::Pair<royale::String, uint64_t>> &registers);

	/**
	* LEVEL 3
	* For each element of the vector a single register read is issued for the connected imager.
	* The second element of each pair will be overwritten by the value of the register given
	* by the first element of the pair :
	*
	* \code
	Vector<Pair<String, uint64_t>> registers;
	registers.push_back (Pair<String, uint64_t> ("0x0B0AD", 0));
	camera->readRegisters (registers);
	\endcode
	*
	* will read out the register 0x0B0AD and will replace the 0 with the current value of
	* the register.
	*
	*
	* \param   registers   Contains elements of possibly not-unique (String, uint64_t) duplets.
	*                      The String component can consist of:
	*                      a) a base-10 decimal number in the range of [0, 65535]
	*                      b) a base-16 hexadecimal number preceded by a "0x" in the
	*                         range of [0, 65535]
	*/
	//PyObject *read_registers(royale::Vector<royale::Pair<royale::String, uint64_t>> &registers);

	/**
	* LEVEL 3
	* Shift the current lens center by the given translation. This works cumulatively (calling
	* shiftLensCenter (0, 1) three times in a row has the same effect as calling shiftLensCenter (0, 3)).
	* If the resulting lens center is not valid this function will return an error.
	* This function works only for raw data readout.
	*
	* \param tx translation in x direction
	* \param ty translation in y direction
	*/
	//PyObject *shift_lens_center(int16_t tx, int16_t ty);

	/**
	* LEVEL 3
	* Retrieves the current lens center.
	*
	* \param x current x center
	* \param y current y center
	*/
	//PyObject *get_lens_center(uint16_t &x, uint16_t &y);

	// -----------------------------------------------------------------------------------------
	// Level 4: Direct imager access (Laser Class 1 not (!) guaranteed) -
	//          activation key required
	// -----------------------------------------------------------------------------------------

	/**
	* LEVEL 4
	* Initialize the camera and configure the system for the specified use case
	*
	* \param initUseCase identifies the use case by an case sensitive string
	*/
	//PyObject *initialize(const royale::String &initUseCase);





};



#endif // !_PYROYALE_H