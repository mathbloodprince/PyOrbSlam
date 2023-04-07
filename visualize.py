import pygame
from pygame.locals import *
from OpenGL.GL import * 
from OpenGL.GLUT import *
from OpenGL.GLU import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glfw
import numpy as np
def camera_pose_rectangle(pose, width, height):
    R, t = pose

    # Half width and height
    hw = width / 2
    hh = height / 2

    # Define the 4 corner points of the rectangle
    corners = np.array([
        [-hw, -hh, 0],
        [hw, -hh, 0],
        [hw, hh, 0],
        [-hw, hh, 0]
    ])

    # Rotate and translate the corner points
    corners_transformed = R @ corners.T + t.reshape(3, 1)
    return corners_transformed.T


def visualize_slam_pygame(poses, points_3D):
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    # Set the window size
    window_size = (width, height)

    # Create a window and set its title
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("SLAM Visualization")

    # Define colors
    red = (255, 0, 0)
    green = (0, 255, 0)
    black = (0, 0, 0)
    
    camera_width = 0.2
    camera_height = 0.2
    scale = 40
    origin = np.array([width // 2, height // 2])
    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        # Clear the screen
        screen.fill(black)

        # Draw camera poses
        for pose in poses:
            corners = camera_pose_rectangle(pose, camera_width, camera_height)
            corners_2d = (corners[:, :2] * scale).astype(int) + origin
            pygame.draw.polygon(screen, red, corners_2d)

        # Draw the point cloud
        for point in points_3D.T:
            x, y, _ = point
            pygame.draw.circle(screen, green, (int(x * 10) + 400, int(y * 10) + 300), 1)

        # Update the display
        pygame.display.flip()

    # Quit pygame
    pygame.quit()


def visualize_slam(poses, points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot camera poses
    xs, ys, zs = [], [], []
    for R, t in poses:
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])
    ax.scatter3D(xs, ys, zs, c='r', marker='o', label='Camera Poses')
    # Plot 3D points
    xs, ys, zs = points_3D
    ax.scatter3D(xs, ys, zs, c='b', marker='^', label='3D Points', s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()



def create_window(width, height, title):
    if not glfw.init():
        return None

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    return window

def draw_points(points, color):
    glColor3f(*color)
    glBegin(GL_POINTS)
    for point in points:
        glVertex3f(*point)
    glEnd()

def draw_camera(pose, width, height, color):
    glColor3f(*color)
    R, t = pose
    hw = width / 2
    hh = height / 2

    corners = np.array([
        [-hw, -hh, 0],
        [hw, -hh, 0],
        [hw, hh, 0],
        [-hw, hh, 0]
    ])

    corners_transformed = R @ corners.T + t.reshape(3, 1)
    corners_transformed = corners_transformed.T

    glBegin(GL_LINE_LOOP)
    for corner in corners_transformed:
        glVertex3f(*corner)
    glEnd()

def visualize_slam_opengl(poses, points_3D):
    width, height = 800, 600
    max_poses, max_plot = 1024//2, 1024
    poses = poses[:max_plot]
    points_3D = points_3D.T[:max_plot]
    
    window = create_window(width, height, "SLAM Visualization")
    if not window:
        return

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    eye = np.array([20, 20, 20])
    center = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    fovy = 600
    near = 0.1
    far = 2000
    
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fovy, width / height, near, far)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*eye, *center, *up)

        draw_points(points_3D, (1.0, 0.0, 0.0))

        camera_width = 1.0 
        camera_height = 1.0
        for pose in poses:
            draw_camera(pose, camera_width, camera_height, (0.0, 1.0, 0.0))

        glfw.swap_buffers(window)

    glfw.terminate()
