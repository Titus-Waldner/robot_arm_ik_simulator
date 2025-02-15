# Robot Arm IK with Auto Drawing

This project simulates a robotic arm using inverse kinematics (IK) in a 3D environment. The arm supports two types of joints (yaw and pitch) and includes interactive controls via mouse input and a GUI (using ImGui). In addition, the program can automatically “draw” a picture by moving the end effector along a path computed on the camera’s view plane. The drawing line is rendered only when it is fully on-screen and always appears on top of the scene.

## Features

- **Inverse Kinematics (IK):**  
  Uses a 3D damped least-squares IK solver to compute the joint angles for the robotic arm.

- **Two Joint Types:**  
  Supports yaw (rotation about the Z-axis) and pitch (rotation about the Y-axis) joints.

- **Interactive Controls:**  
  - **Right-click** (away from the origin) to add a new joint.  
  - **Left-click** near the end effector (or on gizmo arrows) to drag and reposition the arm (manual IK).  
  - **Middle-click** and drag to orbit the camera.

- **Auto Drawing Mode:**  
  When the "Draw Picture" button is pressed, the program computes a square on the camera’s view plane. The arm’s end effector then follows this path and leaves behind particles (simulating "ink"). The connecting line is drawn only when the entire arm is visible on screen and is rendered on top of the scene.

- **Visual Aids:**  
  - Joint markers (displayed as cubes) along the arm, with extra stand geometry for the fixed origin.  
  - Gizmo arrows at the end effector for axis-constrained dragging.

## Dependencies

- [GLFW](https://www.glfw.org/) – For window and input handling.
- [GLAD](https://glad.dav1d.de/) – For OpenGL function loading.
- [GLM](https://glm.g-truc.net/0.9.9/index.html) – For mathematics (vectors, matrices, etc.).
- [Dear ImGui](https://github.com/ocornut/imgui) – For the GUI.
- A C++17 compatible compiler.

## Build Instructions

This project uses CMake for its build system. To build the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/RobotArmIK.git
   cd RobotArmIK
