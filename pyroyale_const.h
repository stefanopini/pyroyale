#ifndef _PYROYALE_CONST_H
#define _PYROYALE_CONST_H


/// helper definition in place of enum classes
// royale::ExposureMode
extern size_t EXPOSURE_MODE_MANUAL;
extern size_t EXPOSURE_MODE_AUTO;
// royale::CameraAccessLevel
extern size_t CAMERA_ACCESS_LEVEL_L1;
extern size_t CAMERA_ACCESS_LEVEL_L2;
extern size_t CAMERA_ACCESS_LEVEL_L3;
extern size_t CAMERA_ACCESS_LEVEL_L4;
// royale::EventSeverity
extern size_t EVENT_SECURITY_ROYALE_INFO;
extern size_t EVENT_SECURITY_ROYALE_WARNING;
extern size_t EVENT_SECURITY_ROYALE_ERROR;
extern size_t EVENT_SECURITY_ROYALE_FATAL;
// royale::EventType
extern size_t EVENT_TYPE_ROYALE_CAPTURE_STREAM;
extern size_t EVENT_TYPE_ROYALE_DEVICE_DISCONNECTED;
extern size_t EVENT_TYPE_ROYALE_OVER_TEMPERATURE;
extern size_t EVENT_TYPE_ROYALE_RAW_FRAME_STATS;


#endif // !_PYROYALE_INTERNALS_H