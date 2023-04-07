import cv2
import numpy as np
from optimize import bundle_adjustment
from visualize import visualize_slam, visualize_slam_pygame, visualize_slam_opengl
import os

def detect_and_compute_orb(frame, orb):
    """Detect orb features"""
    feats = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), maxCorners=1000, qualityLevel=0.001, minDistance=25)
    feats = np.array([cv2.KeyPoint(x=int(f[0][0]), y=int(f[0][1]), size=10) for f in feats])
    kps, des = orb.compute(frame, feats)
    return kps, des

def match_features(des1, des2, matcher):
    """Match features across consecutive frames"""
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def read_video(video_path, W, H):
    """Get all the frames from the video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (W, H)))
    cap.release()
    return frames

def triangulate_points(P1, P2, pts1, pts2):
    """Compute 3D world points from camera poses and kp matches"""
    points_4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_4D /= points_4D[3]
    points_3D = points_4D[:3]
    return points_3D

def find_essential_matrix(matches, kp1, kp2, K):
    """Ransac filter for find essential matrix between kp matches"""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def recover_camera_pose(E, pts1, pts2, K, mask):
    """Estimate camera pose from essential matrix"""
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main(video_path):
    # initialize camera intrinsics matrix
    W, H = 1920 // 2, 1080 //2
    F = 500
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

    # Read video
    frames = read_video(video_path, W, H)

    # Initialize ORB detector and feature matcher
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # initialize a point cloud to accumulate camera poses and 3D point cloud
    point_cloud = []
    poses = [(np.eye(3), np.zeros(3))]
    observations = []
    points_3D = np.empty((3, 0))
    point_correspondences = []
    point_start_indices = [0]

    # Process n_frames
    print("Number of frames: ", len(frames))
    n_frames = 250
    for i in range(n_frames - 1):
        frame1, frame2 = frames[i], frames[i+1]

        # Extract features
        kp1, des1 = detect_and_compute_orb(frame1, orb)
        kp2, des2 = detect_and_compute_orb(frame2, orb)
        
        # Match features and filter
        matches = match_features(des1, des2, matcher)

        # Estimate the essential matrix using matched features
        E, mask = find_essential_matrix(matches, kp1, kp2, K)
        
        # filter the matches based on the essential matrix mask
        filtered_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)
        
        # Recover camera pose using the filtered matches
        R, t = recover_camera_pose(E, pts1, pts2, K, mask)  
        
        # append new camera pose
        poses.append((R, t))
        
        # triangulate matched points to obtain 3D points
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        new_points_3D = triangulate_points(P1, P2, pts1, pts2)
        point_correspondences.append(new_points_3D)
        point_start_indices.append(point_start_indices[-1] + new_points_3D.shape[1])
        
        for idx, m in enumerate(filtered_matches):
            query_idx, train_idx = m.queryIdx, m.trainIdx
            point_idx = point_start_indices[-1] + idx
            observations.append((len(poses) - 1, point_idx, pts1[idx].ravel()))
            observations.append((len(poses) - 2, point_idx, pts2[idx].ravel()))

        # Periodically run bundle adjustment and visualization
        ba_rate = 10 
        if (i + 1) % ba_rate == 0:  # Change this value to control how often bundle adjustment is run
            # Run bundle adjustment
            points_3D = np.hstack(point_correspondences)
            optimized_poses, optimized_points = bundle_adjustment(K, poses, points_3D, observations)

            # Update the map with optimized poses and points
            poses = optimized_poses
            points_3D = optimized_points
            print(f"Optimized at frame: {i + 1}")

    # Visualize the results
    visualize_slam_opengl(optimized_poses, optimized_points)

if __name__ == "__main__":
    path = 'vids/test_countryroad.mp4'
    main(path)
