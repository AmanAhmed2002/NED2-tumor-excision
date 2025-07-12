import numpy as np
import cv2 as cv  # OpenCV library for computer vision
import cv2.aruco as aruco  # ArUco module for marker detection
from pyniryo import *  # Niryo robot library


pi = np.pi


camera_matrix = np.load("mtx.npy")
dist_coeffs = np.load("dist.npy")
new_cam_mtx = np.load("newcameramtx.npy")
roi = np.load("roi.npy")


marker_size = 0.04  # Marker size in meters (e.g., 40 mm)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)  # Use a predefined dictionary for ArUco markers
parameters = cv.aruco.DetectorParameters()  # Detector parameters can be customized if needed
detector = cv.aruco.ArucoDetector(dictionary, parameters)  # Create an ArUco detector with the dictionary/parameters


# Connect to the Niryo NED robot
ned = NiryoRobot("ned2.ee.ryerson.ca")
ned.calibrate_auto()
ned.update_tool()


# Initialize variables for pose estimation
rvecs = []
tvecs = []


c_angle = -20 * np.pi / 180  # Camera tilt angle of -20 degrees (convert to radians)


TC_5 = np.array([
  [np.cos(c_angle), 0, np.sin(c_angle), -0.04],  # Shift of -0.04 m along X-axis
  [0, 1, 0, 0],  # No shift along Y-axis
  [-np.sin(c_angle), 0, np.cos(c_angle), 0],  # No shift along Z-axis
  [0, 0, 0, 1]  # Homogeneous coordinate
])


# Matrix to align the camera's axis orientations
TC_5_C = np.array([
  [0, 1, 0, 0],
  [-1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
], dtype=float)


## WHAT IS THIS BELOW?
T_O_D = np.array([
  [0, -1, 0, -0.03],
  [0, 0, 1, -0.03],
  [-1, 0, 0, 0.055],
  [0, 0, 0, 1]
], dtype=float)


""" Q3.a) """


# # Displacement vector of the entry point (e.g., puncture in the skull) relative to the ArUco marker
# # Students should update these vectors based on the CAD model provided for Q3




d_W_D_to_entry = np.array([-0.12, -0.031, -0.06], dtype=float)


d_W_D_to_tumour = np.array([-0.072, -0.017, -0.078], dtype=float)


d_M_to_entry = np.array([0. - 0.142, -0.0184], dtype=float)


d_M_to_tumour = np.array([-0.0128, -0.1318, -0.0338], dtype=float)


d_W_tumour = np.zeros((3, 1), dtype=np.float64)
d_W_entry = np.zeros((3, 1), dtype=np.float64)


""" The desired offset"""  # !!!!!!!!!!!!!!!!!!!!!!!1
desired_offset = 0.1




P = np.zeros((3, 1), dtype=np.float64)


# Initialize the desired transformation matrix for the robot's end effector
T_W_D = np.eye(4, dtype=float)  # 4x4 identity matrix
pose = []


# Initialize transformation matrices
T_W_C = np.eye(4, dtype=float)  # Transformation from world to camera frame
T_C_M = np.eye(4, dtype=float)  # Transformation from camera to marker frame
T_W_M = np.eye(4, dtype=float)  # Transformation from world to marker frame
T_W_EE = np.eye(4, dtype=float)  # Transformation from world to end effector frame


# Define unit vectors along the x, y, z axes
unit_vectors = np.array([
  [1, 0, 0],  # x-axis
  [0, 1, 0],  # y-axis
  [0, 0, 1]  # z-axis
])




def euler_to_rot_matrix(roll, pitch, yaw):


  # Rotation matrix around x-axis
  Rx = np.array([
      [1, 0, 0],
      [0, np.cos(roll), -np.sin(roll)],
      [0, np.sin(roll), np.cos(roll)]
  ])


  # Rotation matrix around y-axis
  Ry = np.array([
      [np.cos(pitch), 0, np.sin(pitch)],
      [0, 1, 0],
      [-np.sin(pitch), 0, np.cos(pitch)]
  ])


  # Rotation matrix around z-axis
  Rz = np.array([
      [np.cos(yaw), -np.sin(yaw), 0],
      [np.sin(yaw), np.cos(yaw), 0],
      [0, 0, 1]
  ])


  # Combined rotation matrix
  R = Rz @ Ry @ Rx
  # R = np.dot(Rz, np.dot(Ry, Rx))
  return R




def closest_angle(angle, initial_angle):
  """
  Finds the angle closest to the initial_angle within a range of -2π to 2π.
  """
  diff = angle - initial_angle
  # Normalize to the range [-2π, 2π]
  diff = (diff + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
  closest_angle = initial_angle + diff
  return closest_angle




def R_to_roll_pitch_yaw(R, initial_roll, initial_pitch, initial_yaw):


  assert R.shape == (3, 3), "Matrix must be 3x3"


  U, _, Vt = np.linalg.svd(R)
  # R = np.dot(U, Vt)
  R = U @ Vt


  # Transformation for NED (North, East, Down) convention (if necessary)
  T = np.array([
      [0, 0, 1],
      [0, -1, 0],
      [-1, 0, 0]
  ])


  # Adjust rotation matrix for NED convention
  R_NED = T @ R @ T.T


  # Calculate Euler angles from the transformed rotation matrix
  roll = np.arctan2(R_NED[1, 2], R_NED[2, 2])
  pitch = np.arctan2(-R_NED[0, 2], np.sqrt(R_NED[1, 2] ** 2 + R_NED[2, 2] ** 2))
  yaw = np.arctan2(R_NED[0, 1], R_NED[0, 0])


  # Adjust angles to be close to initial values
  roll = closest_angle(roll, initial_roll)
  pitch = closest_angle(pitch, initial_pitch)
  yaw = closest_angle(yaw, initial_yaw)


  return roll, pitch, yaw




def Estimate(corners, marker_size, camera_matrix, distortion):


  marker_points = np.array([
      [-marker_size / 2, marker_size / 2, 0],  # Top-left corner
      [marker_size / 2, marker_size / 2, 0],  # Top-right corner
      [marker_size / 2, -marker_size / 2, 0],  # Bottom-right corner
      [-marker_size / 2, -marker_size / 2, 0]  # Bottom-left corner
  ], dtype=np.float32)


  success_list = []
  rvecs = []
  tvecs = []


  for c in corners:
      c = np.array(c, dtype=np.float32)  # Ensure data type is correct
      success, rvec, tvec = cv.solvePnP(marker_points, c, camera_matrix, distortion, flags=cv.SOLVEPNP_IPPE_SQUARE)
      rvecs.append(rvec)
      tvecs.append(tvec)
      success_list.append(success)


  return rvecs, tvecs, success_list




def rotate_vector(vector, axis, angle):


  return (
          vector * np.cos(angle) +
          np.cross(axis, vector) * np.sin(angle) +
          axis * np.dot(axis, vector) * (1 - np.cos(angle))
  )




def Get_Object_to_Camera_Pose(marker_size, camera_matrix, dist_coeffs):


  # Capture image from the robot's camera
  rvecs = []
  tvecs = []
  img_compressed = ned.get_img_compressed()
  img = uncompress_image(img_compressed)
  x, y, w, h = roi
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale


  # Display the live stream (optional)
  cv.imshow("Live Stream", gray)


  # Detect ArUco markers in the image
  markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)


  # Refine detected markers
  if markerIds is not None:
      # Estimate the poses of the detected markers
      rvecs, tvecs, _ = Estimate(markerCorners, marker_size, camera_matrix, dist_coeffs)


      for i in range(len(markerIds)):
          # Draw axis for each marker
          cv.drawFrameAxes(gray, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.02)
      # Draw the detected markers
      aruco.drawDetectedMarkers(gray, markerCorners, markerIds)


  # Display the processed frame (optional)
  cv.imshow('Processed Frame', gray)


  # Initialize the transformation matrix
  T_C_M = np.eye(4, dtype=float)


  if not rvecs:
      print("No markers detected.")
  else:
      for idx, rvecs in enumerate(rvecs):
          # Check if any rvec is a zero vector
          if np.all(rvecs == 0):
              print(f"rvec at index {idx} is a zero vector.")
          else:
              # Process non-zero rvec
              R_mtx, _ = cv.Rodrigues(rvecs)
              print(f"Translation vector for tvec at index {idx}: \n{tvecs}")
      T_C_M[0:3, 0:3] = R_mtx  # Creating T_O_C and adding rotation part
      for j in range(3):
          T_C_M[j, 3] = tvecs[0][j].item()  # Creating T_C_M and adding translation part


  return T_C_M




def A_mtx(a, alpha, d, theta):


  A = np.array([
      [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
      [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
      [0, np.sin(alpha), np.cos(alpha), d],
      [0, 0, 0, 1]
  ])
  return A




def calculate_R_given_x(x_new, R_EE):


  x_new = x_new[:, 0] if x_new.ndim > 1 else x_new  # Making sure x_new is a 1D array


  # Old rotation matrix and axes
  x_old = R_EE[:, 0]
  y_old = R_EE[:, 1]
  z_old = R_EE[:, 2]


  # Calculate rotation axis and angle between old and new x-axis
  rot_axis = np.cross(x_old, x_new)
  rot_axis /= np.linalg.norm(rot_axis)  # Normalize the rotation axis
  rot_angle = np.arccos(np.dot(x_old, x_new))


  # Rotate old axes to get new axes
  y_new = rotate_vector(y_old, rot_axis, rot_angle)
  z_new = rotate_vector(z_old, rot_axis, rot_angle)


  # Normalize y_new and z_new
  y_new /= np.linalg.norm(y_new)
  z_new /= np.linalg.norm(z_new)


  # Ensure orthogonality
  z_new = np.cross(x_new, y_new)
  z_new /= np.linalg.norm(z_new)
  y_new = np.cross(z_new, x_new)
  y_new /= np.linalg.norm(y_new)


  # Construct the new rotation matrix
  R_new = np.column_stack((x_new, y_new, z_new))
  return R_new




'''Main Execution Loop'''
j = 0


while j < 200:
  j += 1
  # Get the transformation from camera to marker frame
  T_C_M = Get_Object_to_Camera_Pose(marker_size, camera_matrix, dist_coeffs)
  ###########################################################################################
  ''' Q1. Students to calculate T_W_C '''


  # Getting Camera coordinate relative to world
  q = ned.get_joints()  # Get current joint angles from the robot


  a = [0, .221, 0.0282, 0, 0, 0.035]
  alpha = [np.pi / 2, 0, np.pi / 2, -(np.pi / 2), np.pi / 2, 0]
  d = [0.183, 0, 0, 0.235, 0, 0.06]
  theta = [q[0], (q[1] + (np.pi / 2)), q[2], q[3] + (np.pi), -q[4], (q[5] + (np.pi / 2))]
  # T_W_5 = DH(a, alpha, d, theta, 5)
  A1 = A_mtx(a[0], alpha[0], d[0], theta[0])
  A2 = A_mtx(a[1], alpha[1], d[1], theta[1])
  A3 = A_mtx(a[2], alpha[2], d[2], theta[2])
  A4 = A_mtx(a[3], alpha[3], d[3], theta[3])
  A5 = A_mtx(a[4], alpha[4], d[4], theta[4])
  T_5 = A1 @ A2 @ A3 @ A4 @ A5


  # transformation matrix of camera relative to world frame
  T_C_W = T_5 @ TC_5 @ TC_5_C
  ''' End Q1'''
  ###########################################################################################
  '''Q2. Students need to calculate T_W_M '''


  T_M_W = T_C_W @ T_C_M  # Transformation from world to marker frame
  print('T_W_M', T_M_W)
  '''End Q2'''


  #########################################################
  # Get the robot's current pose (position and orientation)
  pose = ned.pose


  # Build the transformation matrix for the end effector
  T_W_EE = np.eye(4)
  T_W_EE[0:3, 3] = [pose.x, pose.y, pose.z]  # Position


  EE_rot_vector = np.array([pose.roll, pose.pitch, pose.yaw], dtype=float)
  R_EE = euler_to_rot_matrix(pose.roll, pose.pitch, pose.yaw)  # Orientation (Euler to R)
  T_W_EE[0:3, 0:3] = R_EE  # Implementing R_EE in homogeneous transformation for EE rel. to World
  ###########################################################################################
  ''' Q3. Students need to compute d_W_tumour and d_W_entry '''


  R_W_M = T_M_W[:3, :3]
  P_W_M = T_M_W[:3, 3]


  T_W_D = T_M_W @ T_O_D


  d_W_entry[0] = T_W_D[0, 3] + d_W_D_to_entry[0]
  d_W_entry[1] = T_W_D[1, 3] + d_W_D_to_entry[1]
  d_W_entry[2] = T_W_D[2, 3] + d_W_D_to_entry[2]


  d_W_tumour[0] = T_W_D[0, 3] + d_W_D_to_tumour[0]
  d_W_tumour[1] = T_W_D[1, 3] + d_W_D_to_tumour[1]
  d_W_tumour[2] = T_W_D[2, 3] + d_W_D_to_tumour[2]


  print('d_W_entry=', d_W_entry)
  print('d_W_tumour=', d_W_tumour)


  '''End of Q3'''
  ###########################################################################################
  '''Q4/Q5. Students need to calculate the desired transformation (orientation/position)'''


  x_W_entry_to_tumor = d_W_entry - d_W_tumour


  x_new = - x_W_entry_to_tumor
  x_new = x_new / np.linalg.norm(x_new)


  # Compute the new rotation matrix
  R_new = calculate_R_given_x(x_new, R_EE)


  # Compute the desired position at a specified offset along the trajectory
  P = np.reshape(d_W_entry, (3,)) + desired_offset * R_new[0:3, 0]


  # Update the desired transformation matrix
  T_W_D[:3, :3] = R_new
  T_W_D[:3, 3] = P
  print('T_W_D', T_W_D)


  """ End of Q4/Q5. """
  ###########################################################################################
  # Convert the desired rotation matrix to Euler angles
  roll_W_D, pitch_W_D, yaw_W_D = R_to_roll_pitch_yaw(T_W_D[0:3, 0:3], pose.roll, pose.pitch, pose.yaw)


  # Create a PoseObject with the desired pose
  place_pose = PoseObject(
      x=T_W_D[0, 3], y=T_W_D[1, 3], z=T_W_D[2, 3],
      roll=roll_W_D, pitch=pitch_W_D, yaw=yaw_W_D
  )


  print(place_pose)  # Print the desired pose
  ned.move_pose(place_pose)  # Move the robot to the desired pose


ned.close_connection()
cv.destroyAllWindows()
