# PyOrbSlam
Orb feature based SLAM (simultaneous localization and mapping) using OpenCV for feature extraction and pose estimation, g2o for bundle adjustment, and OpenGL for visualization.

Note:
This was performed on macOS by installing g2opy with: pip install g2o-python.  The original g2opy is buggy in downloading with mac and the syntax in this g2opy version I used may be different when performing bundle adjustment.

To run the code: python orb_slam.py

References from other monocular orb based SLAM implementations:
https://github.com/geohot/twitchslam
https://github.com/Akbonline/SLAMPy-Monocular-SLAM-implementation-in-Python/blob/master/slam.py

