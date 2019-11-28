import pyroyale as royale
import numpy as np
from matplotlib import pyplot as plt
import time


print('imported')

manager = royale.PyCameraManager()
manager.initialize()

print('manager ok')

cameras = manager.get_connected_cameras()
print(cameras)
#camera = royale.PyCameraDevice(h_camera)
camera = manager.create_camera_device(cameras[0])
camera.initialize()

print('camera ok')

print(camera.get_camera_info())
print(camera.get_camera_name())
print(camera.get_use_cases())
print(camera.get_current_use_case())
#print(camera.get_exposure_mode())
print(camera.get_exposure_mode(0))
#print(camera.get_exposure_mode(1))

camera.set_use_case(camera.get_use_cases()[2])

print(camera.get_current_use_case())

counter = -1
counter2 = -1
counter3 = -1
counter4 = -1

def callback(timestamp, stream_id, width, height, points, gray_image, depth_confidence):
	global counter
	counter += 1
	print('Data received', points.shape, gray_image.shape, depth_confidence.shape)
	img = np.array(gray_image / 2**8, dtype=np.uint8).copy()
	plt.imsave('./imgs/img_%2d.png' % counter, img)
	return

def callback2(timestamp, stream_id, width, height, depth, depth_confidence):
	global counter2
	counter2 += 1
	print('Data received 2', depth_image.shape, depth_confidence.shape)
	img = np.array(depth_image / 2**5, dtype=np.uint8).copy()
	plt.imsave('./imgs/img_depth_%2d.png' % counter2, img)
	return

def callback3(timestamp, stream_id, width, height, ir_image):
	global counter3
	counter3 += 1
	print('Data received 3', ir_image.shape)
	img = np.array(ir_image, dtype=np.uint8).copy()
	plt.imsave('./imgs/img_ir_%2d.png' % counter3, img)
	return

def callback4(timestamp, stream_id, num_points, sparse_point_cloud):
	global counter4
	counter4 += 1
	print('Data received 4', sparse_point_cloud.shape)
	#print(sparse_point_cloud[0:4])
	# ToDo Save pointcloud
	return

camera.register_data_listener(callback)
camera.register_depth_image_listener(callback2)
camera.register_ir_image_listener(callback3)
camera.register_sparse_point_cloud_listener(callback4)

#
camera.start_capture()

print('is_capturing: ', camera.is_capturing())

time.sleep(10)

camera.stop_capture()
#

print('done')
print('is_capturing: ', camera.is_capturing())

time.sleep(1)
camera.unregister_data_listener()
camera.unregister_depth_image_listener()
camera.unregister_ir_image_listener()

print('end')
