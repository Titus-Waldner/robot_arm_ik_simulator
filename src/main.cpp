#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Dear ImGui (assumed to be integrated with GLFW/OpenGL3 backends)
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// ----- Window Dimensions -----
const unsigned int WIDTH = 800, HEIGHT = 600;

// Forward declaration:
glm::vec3 computeCameraPos();

// ----- Shader Sources -----
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main(){
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main(){
    FragColor = uColor;
}
)";

// ----- Geometry -----
// A unit cube (centered at the origin, size 1) with 36 vertices.
float cubeVertices[] = {
    // Back face
    -0.5f, -0.5f, -0.5f,    0.5f, -0.5f, -0.5f,    0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,   -0.5f,  0.5f, -0.5f,   -0.5f, -0.5f, -0.5f,
    // Front face
    -0.5f, -0.5f,  0.5f,    0.5f, -0.5f,  0.5f,    0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,   -0.5f,  0.5f,  0.5f,   -0.5f, -0.5f,  0.5f,
    // Left face
    -0.5f,  0.5f,  0.5f,   -0.5f,  0.5f, -0.5f,   -0.5f, -0.5f, -0.5f,
   -0.5f, -0.5f, -0.5f,   -0.5f, -0.5f,  0.5f,   -0.5f,  0.5f,  0.5f,
    // Right face
     0.5f,  0.5f,  0.5f,    0.5f,  0.5f, -0.5f,    0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,    0.5f, -0.5f,  0.5f,    0.5f,  0.5f,  0.5f,
    // Bottom face
    -0.5f, -0.5f, -0.5f,    0.5f, -0.5f, -0.5f,    0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,   -0.5f, -0.5f,  0.5f,   -0.5f, -0.5f, -0.5f,
    // Top face
    -0.5f,  0.5f, -0.5f,    0.5f,  0.5f, -0.5f,    0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,   -0.5f,  0.5f,  0.5f,   -0.5f,  0.5f, -0.5f
};

// ----- Joint Data -----
// Only two types of moveable joints are allowed: YAW and PITCH.
// The fixed origin cube (drawn with extra stand geometry) is not part of the IK chain.
enum JointType { YAW, PITCH };

struct Joint {
    glm::vec3 axis;  // Rotation axis (in the joint’s local frame)
    float angle;     // Angle (in radians)
    JointType type;
};

std::vector<Joint> joints;  // Contains only moveable joints (origin remains fixed)

JointType uiSelectedJoint = YAW; // Default type for new joints

// Each added joint translates along its local X by a fixed length.
const float linkLength = 1.0f;

// ----- Forward Kinematics (FK) -----
// The first position is always the fixed origin (at (0,0,0)).
// Then, for each moveable joint, we apply its rotation and then translate along its local X.
std::vector<glm::vec3> computeFK() {
    std::vector<glm::vec3> positions;
    glm::mat4 T = glm::mat4(1.0f);
    positions.push_back(glm::vec3(T[3])); // fixed origin
    for (size_t i = 0; i < joints.size(); i++) {
        T = T * glm::rotate(glm::mat4(1.0f), joints[i].angle, joints[i].axis)
              * glm::translate(glm::mat4(1.0f), glm::vec3(linkLength, 0, 0));
        positions.push_back(glm::vec3(T[3]));
    }
    return positions;
}

// ----- Maximum Reach Clamping -----
float maxReach() {
    return joints.size() * linkLength;
}

glm::vec3 clampTarget(const glm::vec3 &target) {
    float reach = maxReach();
    if (glm::length(target) > reach)
        return glm::normalize(target) * reach;
    return target;
}

// ----- IK Solver (Damped Least-Squares) -----
// Adjusts only the moveable joints.
void solveIK3D(const glm::vec3 &rawTarget) {
    if(joints.empty()) return;
    glm::vec3 target = clampTarget(rawTarget);
    int n = joints.size();
    const float lambda = 0.1f;
    const int iterations = 10;
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<glm::vec3> positions;
        std::vector<glm::mat4> Ts;
        glm::mat4 T = glm::mat4(1.0f);
        positions.push_back(glm::vec3(T[3]));  // fixed origin
        Ts.push_back(T);
        for (int i = 0; i < n; i++) {
            T = T * glm::rotate(glm::mat4(1.0f), joints[i].angle, joints[i].axis)
                  * glm::translate(glm::mat4(1.0f), glm::vec3(linkLength, 0, 0));
            positions.push_back(glm::vec3(T[3]));
            Ts.push_back(T);
        }
        glm::vec3 p_end = positions.back();
        glm::vec3 error = target - p_end;
        if (glm::length(error) < 0.001f)
            break;
        glm::mat3 A(0.0f);
        std::vector<glm::vec3> J;
        for (int i = 0; i < n; i++) {
            glm::vec3 w = glm::normalize(glm::vec3(Ts[i] * glm::vec4(joints[i].axis, 0.0f)));
            glm::vec3 diff = p_end - positions[i];
            glm::vec3 Ji = glm::cross(w, diff);
            J.push_back(Ji);
            A += glm::outerProduct(Ji, Ji);
        }
        A += lambda * lambda * glm::mat3(1.0f);
        glm::mat3 invA = glm::inverse(A);
        for (int i = 0; i < n; i++) {
            float delta = glm::dot(J[i], invA * error);
            joints[i].angle += delta;
        }
    }
}

// ----- Automatic Drawing (on the Camera's View Plane) -----
// When the "Draw Picture" button is pressed, we compute a square on the current view plane
// (so it appears to cover your monitor). The end effector then automatically moves along that square.
bool autoDrawing = false;
std::vector<glm::vec3> drawingPath;

int drawingPathIndex = 0;
// Increase drawing speed for a snappier movement.
float drawingSpeed = 2.0f; // units per second

// Global variables to control particle fading. While autoDrawing is active, particles are opaque.
// When drawing completes, we store the finish time and start fading all particles uniformly.
bool drawingFinished = false;
float drawingFinishTime = 0.0f;

// Compute a square on the view plane of the camera. The square covers the smaller of the view dimensions.
void initDrawingPath() {
    if(joints.empty()) return; // need at least one moveable joint

    // Get the current end effector position.
    std::vector<glm::vec3> positions = computeFK();
    glm::vec3 ee = positions.back();

    // Get the camera’s current position and basis.
    glm::vec3 camPos = computeCameraPos();
    glm::vec3 viewDir = glm::normalize(glm::vec3(0,0,0) - camPos); // camera looks toward origin
    glm::vec3 up = glm::vec3(0,0,1);
    glm::vec3 right = glm::normalize(glm::cross(viewDir, up));
    up = glm::normalize(glm::cross(right, viewDir));

    // Choose a plane at distance d from the camera (at origin's depth).
    float d = glm::length(camPos);
    glm::vec3 center = camPos + viewDir * d;

    // Compute a scale factor S from the vertical FOV.
    float fov = glm::radians(45.0f);
    float height = 2.0f * d * tan(fov/2.0f);
    float aspect = (float)WIDTH / HEIGHT;
    float width = height * aspect;
    float S = std::min(width, height);

    // ====================================================
    // Define a heart shape using the standard parametric equations:
    //    x(t) = 16 sin^3(t)
    //    y(t) = 13 cos(t) - 5 cos(2t) - 2 cos(3t) - cos(4t)
    // t in [0, 2pi].
    //
    // The original heart spans roughly from y = -17 to y = 5,
    // so its vertical center is about -6.
    // To center it on the origin, we add 6 to y.
    //
    // We also use a smaller scaleFactor so the heart is smaller.
    // ====================================================
    std::vector<glm::vec2> heartPath;
    const int numSegments = 100;
    float scaleFactor = 0.015f;  // smaller than before for a smaller heart
    for (int i = 0; i <= numSegments; i++) {
        float t = (float)i / numSegments * 2.0f * glm::pi<float>();
        float x = 16.0f * pow(sin(t), 3);
        float y = 13.0f * cos(t) - 5.0f * cos(2*t) - 2.0f * cos(3*t) - cos(4*t);
        // Add 6 to y so that the heart is shifted upward (centering its vertical extent)
        heartPath.push_back(glm::vec2(x * scaleFactor, (y + 6.0f) * scaleFactor));
    }

    // ====================================================
    // Map our 2D heart coordinates into world-space points on the view plane.
    // We interpret the normalized x coordinate as a multiplier for the "right" vector
    // and the normalized y coordinate as a multiplier for the "up" vector.
    // The factor S scales our normalized units to the size of the view.
    // ====================================================
    drawingPath.clear();
    // Start the drawing path at the current end-effector position so that no trail is drawn while moving.
    drawingPath.push_back(ee);
    for (const auto &pt : heartPath) {
        glm::vec3 worldPoint = center + (pt.x * S) * right + (pt.y * S) * up;
        drawingPath.push_back(worldPoint);
    }
    
    drawingPathIndex = 1; // begin moving toward the first target
    autoDrawing = true;
    drawingFinished = false;
    drawingFinishTime = 0.0f;
}



// ----- Orbit Camera Controls -----
// We use a world up of (0,0,1) and spherical coordinates.
float cameraDistance = 8.0f;
float cameraYaw   = glm::radians(45.0f);   // azimuth
float cameraPitch = glm::radians(30.0f);     // elevation

glm::vec3 computeCameraPos() {
    float x = cameraDistance * cos(cameraPitch) * cos(cameraYaw);
    float y = cameraDistance * cos(cameraPitch) * sin(cameraYaw);
    float z = cameraDistance * sin(cameraPitch);
    return glm::vec3(x, y, z);
}

glm::mat4 view;
void updateView(glm::mat4 &viewMatrix) {
    glm::vec3 camPos = computeCameraPos();
    viewMatrix = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0,0,1));
}

glm::mat4 projection;

// ----- Gizmo for Axis Dragging -----
// Three arrows (along world X, Y, Z) drawn at the end effector.
struct Arrow {
    glm::vec3 direction;
    glm::vec3 color;
};
std::vector<Arrow> gizmoArrows = {
    { glm::vec3(1,0,0), glm::vec3(1,0,0) },
    { glm::vec3(0,1,0), glm::vec3(0,1,0) },
    { glm::vec3(0,0,1), glm::vec3(0,0,1) }
};
const float arrowLength = 0.5f;

// ----- Global Variables for Mouse Interaction -----
int constrainedAxis = -1;
bool draggingEndEffector = false;
bool rotatingCamera = false;
double lastMouseX = 0.0, lastMouseY = 0.0;

// ----- Particle System -----
// Each particle is a small cube. During drawing the particles are opaque;
// when drawing finishes, they all fade over 30 seconds.
struct Particle {
    glm::vec3 pos;
};
std::vector<Particle> particles;

// ----- Mouse Callbacks -----
// (User input is ignored while autoDrawing is active.)
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if(autoDrawing) return;
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    
    std::vector<glm::vec3> positions = computeFK();
    glm::vec3 ee = positions.back();
    glm::vec4 clipPos = projection * view * glm::vec4(ee, 1.0f);
    glm::vec3 ndc = glm::vec3(clipPos) / clipPos.w;
    float screenEE_X = (ndc.x * 0.5f + 0.5f) * w;
    float screenEE_Y = (1.0f - (ndc.y * 0.5f + 0.5f)) * h;
    glm::vec2 eeScreen(screenEE_X, screenEE_Y);
    
    glm::vec2 mousePos(mouseX, mouseY);
    float distToEE = glm::length(mousePos - eeScreen);
    const float pickThreshold = 20.0f;
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS && positions.size() > 1) {
            int axisConstraint = -1;
            float bestDist = pickThreshold;
            for (int i = 0; i < 3; i++) {
                glm::vec3 arrowTipWorld = ee + arrowLength * gizmoArrows[i].direction;
                glm::vec4 tipClip = projection * view * glm::vec4(arrowTipWorld, 1.0f);
                glm::vec3 tipNDC = glm::vec3(tipClip) / tipClip.w;
                float tipScreenX = (tipNDC.x * 0.5f + 0.5f) * w;
                float tipScreenY = (1.0f - (tipNDC.y * 0.5f + 0.5f)) * h;
                glm::vec2 tipScreen(tipScreenX, tipScreenY);
                float d = glm::length(mousePos - tipScreen);
                if(d < bestDist) {
                    bestDist = d;
                    axisConstraint = i;
                }
            }
            if (axisConstraint != -1 || distToEE < pickThreshold) {
                draggingEndEffector = true;
                constrainedAxis = axisConstraint; // -1 indicates free drag
                lastMouseX = mouseX;
                lastMouseY = mouseY;
            }
        } else if (action == GLFW_RELEASE) {
            draggingEndEffector = false;
            constrainedAxis = -1;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        if (distToEE >= pickThreshold) {
            glm::vec3 newAxis;
            JointType newType = uiSelectedJoint;
            if(newType == YAW)
                newAxis = glm::vec3(0,0,1);
            else
                newAxis = glm::vec3(0,1,0);
            joints.push_back({ newAxis, 0.0f, newType });
        }
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            rotatingCamera = true;
            lastMouseX = mouseX;
            lastMouseY = mouseY;
        } else if (action == GLFW_RELEASE) {
            rotatingCamera = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if(!autoDrawing && draggingEndEffector) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        std::vector<glm::vec3> positions = computeFK();
        glm::vec3 ee = positions.back();
        glm::vec4 clipPos = projection * view * glm::vec4(ee, 1.0f);
        float depth = (clipPos.z/clipPos.w + 1.0f) / 2.0f;
        glm::vec4 viewport(0,0,w,h);
        glm::vec3 winCoord(xpos, h - ypos, depth);
        glm::vec3 target = glm::unProject(winCoord, view, projection, viewport);
        if(constrainedAxis != -1) {
            glm::vec3 dir;
            if(constrainedAxis == 0)      dir = glm::vec3(1,0,0);
            else if(constrainedAxis == 1) dir = glm::vec3(0,1,0);
            else                          dir = glm::vec3(0,0,1);
            glm::vec3 diff = target - ee;
            float projLength = glm::dot(diff, dir);
            target = ee + projLength * dir;
        }
        solveIK3D(target);
    }
    if(rotatingCamera) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        lastMouseX = xpos;
        lastMouseY = ypos;
        float sensitivity = 0.005f;
        cameraYaw   += sensitivity * dx;
        cameraPitch += sensitivity * dy;
        cameraPitch = glm::clamp(cameraPitch, glm::radians(-89.0f), glm::radians(89.0f));
        updateView(view);
    }
}

// ----- Shader Compilation Helper -----
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

// ----- Main -----
int main(){
    if(!glfwInit()){
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Robot Arm IK with Fixed Origin, Stand & Auto Drawing on View Plane", nullptr, nullptr);
    if(!window){ glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to load GLAD\n";
        return -1;
    }
    
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    
    // Setup ImGui.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window,true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();
    
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    
    int colorLoc = glGetUniformLocation(shaderProgram, "uColor");
    
    // Setup geometry: cube for joint markers.
    GLuint cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Setup geometry: line for drawing the arm.
    GLuint lineVAO, lineVBO;
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f,0.1f,0.1f,1.0f);
    
    updateView(view);
    projection = glm::perspective(glm::radians(45.0f), (float)WIDTH/HEIGHT, 0.1f, 100.0f);
    
    int modelLoc = glGetUniformLocation(shaderProgram, "model");
    int viewLoc  = glGetUniformLocation(shaderProgram, "view");
    int projLoc  = glGetUniformLocation(shaderProgram, "projection");
    
    // Timing for animation.
    float lastFrameTime = (float)glfwGetTime();
    
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
        int winW, winH;
        glfwGetFramebufferSize(window, &winW, &winH);
        projection = glm::perspective(glm::radians(45.0f), (float)winW/winH, 0.1f, 100.0f);
        
        float currentTime = (float)glfwGetTime();
        float dt = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
		// --- Automatic Drawing Update ---
		if(autoDrawing && !drawingPath.empty() && drawingPathIndex < drawingPath.size()) {
			std::vector<glm::vec3> positions = computeFK();
			glm::vec3 ee = positions.back();
			glm::vec3 targetPoint = drawingPath[drawingPathIndex];
			glm::vec3 diff = targetPoint - ee;
			float dist = glm::length(diff);
			glm::vec3 newTarget = ee;
			
			// Set a relaxed threshold so that if the distance is small enough, we snap to the target.
			const float threshold = 0.01f;
			float step = drawingSpeed * dt;
			
			if(dist < threshold) {
				// Close enough: snap to the target.
				newTarget = targetPoint;
				if(drawingPathIndex > 1)
					particles.push_back({ targetPoint });
				drawingPathIndex++;
				if(drawingPathIndex >= drawingPath.size()) {
					autoDrawing = false;
					drawingFinished = true;
					drawingFinishTime = currentTime;
				}
			} else if(step >= dist) {
				// If the step is enough to cover the remaining distance.
				newTarget = targetPoint;
				if(drawingPathIndex > 1)
					particles.push_back({ targetPoint });
				drawingPathIndex++;
				if(drawingPathIndex >= drawingPath.size()) {
					autoDrawing = false;
					drawingFinished = true;
					drawingFinishTime = currentTime;
				}
			} else {
				// Otherwise, take a normal step toward the target.
				newTarget = ee + (diff/dist) * step;
			}
			
			solveIK3D(newTarget);
			// Drop a particle each frame if we are in drawing mode.
			if(drawingPathIndex > 1)
				particles.push_back({ ee });
		}

        
        // --- Particle Update ---
        // If drawing is finished, particles start fading (all use the same fade timer).
        float particleAlpha = 1.0f;
        if(drawingFinished) {
            float fade = (currentTime - drawingFinishTime) / 30.0f;
            particleAlpha = std::max(1.0f - fade, 0.0f);
        }
        // For simplicity, remove particles once fully faded.
        for (size_t i = 0; i < particles.size(); ) {
            // When drawing is finished, we remove all particles after 30 sec.
            if(drawingFinished && (currentTime - drawingFinishTime) >= 30.0f)
                particles.erase(particles.begin() + i);
            else
                i++;
        }
        
        // --- UI ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::Begin("Robot Arm Controls");
        if(ImGui::Button("Remove Last Joint") && !joints.empty())
            joints.pop_back();
        const char* jointOptions[] = { "Yaw (rotate about Z, moves in XY)", "Pitch (rotate about Y, moves in XZ)" };
        int currentOption = (uiSelectedJoint == YAW) ? 0 : 1;
        if(ImGui::Combo("New Joint Type", &currentOption, jointOptions, IM_ARRAYSIZE(jointOptions)))
            uiSelectedJoint = (currentOption == 0) ? YAW : PITCH;
        ImGui::Text("Right-click (away from the origin) to add a new joint.");
        ImGui::Text("Left-click near the end effector (or gizmo arrows) and drag to move manually.");
        ImGui::Text("Middle-click and drag to orbit the camera.");
        if(ImGui::Button("Draw Picture")) {
            if(!joints.empty()) {
                initDrawingPath();
                drawingFinished = false; // reset fade timer
            }
        }
        ImGui::End();
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        
        // --- Draw the Arm ---
        std::vector<glm::vec3> positions = computeFK();
        std::vector<float> lineVerts;
        for(auto &p : positions) {
            lineVerts.push_back(p.x);
            lineVerts.push_back(p.y);
            lineVerts.push_back(p.z);
        }
        glBindVertexArray(lineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
        glBufferData(GL_ARRAY_BUFFER, lineVerts.size()*sizeof(float), lineVerts.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glUniform4f(colorLoc, 1.0f, 0.6f, 0.2f, 1.0f); // orange
        glm::mat4 ident = glm::mat4(1.0f);
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(ident));
        glLineWidth(3.0f);
        glDrawArrays(GL_LINE_STRIP, 0, positions.size());
        
        // --- Draw Joint Markers ---
        for(size_t i = 0; i < positions.size(); i++){
            glm::mat4 model = glm::translate(glm::mat4(1.0f), positions[i]);
            model = glm::scale(model, glm::vec3(0.2f));
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            if(i == 0)
                glUniform4f(colorLoc, 0.1f, 0.1f, 0.6f, 1.0f); // fixed origin (dark blue)
            else if(i == positions.size()-1)
                glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // end effector (white)
            else
                glUniform4f(colorLoc, 0.2f, 0.2f, 1.0f, 1.0f);
            glBindVertexArray(cubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            
            // For the fixed origin (index 0), draw the stand.
            if(i == 0) {
                glm::vec3 basePos = positions[0];
                glm::vec3 poleStartLocal = glm::vec3(0, 0, -0.1f);
                glm::vec3 poleEndLocal   = glm::vec3(0, 0, -0.6f);
                glm::vec3 poleStart = basePos + poleStartLocal;
                glm::vec3 poleEnd   = basePos + poleEndLocal;
                std::vector<float> poleVerts = {
                    poleStart.x, poleStart.y, poleStart.z,
                    poleEnd.x,   poleEnd.y,   poleEnd.z
                };
                GLuint poleVAO, poleVBO;
                glGenVertexArrays(1, &poleVAO);
                glGenBuffers(1, &poleVBO);
                glBindVertexArray(poleVAO);
                glBindBuffer(GL_ARRAY_BUFFER, poleVBO);
                glBufferData(GL_ARRAY_BUFFER, poleVerts.size() * sizeof(float), poleVerts.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
                glEnableVertexAttribArray(0);
                glUniform4f(colorLoc, 0.8f, 0.8f, 0.8f, 1.0f);
                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(ident));
                glLineWidth(2.0f);
                glDrawArrays(GL_LINES, 0, 2);
                glDeleteVertexArrays(1, &poleVAO);
                glDeleteBuffers(1, &poleVBO);
                
                float cupRadius = 0.3f;
                int circleSegments = 40;
                int numCircles = 6;
                for (int ci = 1; ci < numCircles; ci++) {
                    float theta = (float)ci / (numCircles - 1) * (glm::pi<float>() / 2.0f);
                    float currCircleRadius = cupRadius * sin(theta);
                    float circleZ = -cupRadius * (1 - cos(theta));
                    std::vector<float> circleVerts;
                    for (int j = 0; j < circleSegments; j++) {
                        float phi = 2.0f * glm::pi<float>() * j / circleSegments;
                        float x = currCircleRadius * cos(phi);
                        float y = currCircleRadius * sin(phi);
                        float z = circleZ;
                        circleVerts.push_back(poleEnd.x + x);
                        circleVerts.push_back(poleEnd.y + y);
                        circleVerts.push_back(poleEnd.z + z);
                    }
                    GLuint circleVAO, circleVBO;
                    glGenVertexArrays(1, &circleVAO);
                    glGenBuffers(1, &circleVBO);
                    glBindVertexArray(circleVAO);
                    glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
                    glBufferData(GL_ARRAY_BUFFER, circleVerts.size()*sizeof(float), circleVerts.data(), GL_STATIC_DRAW);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
                    glEnableVertexAttribArray(0);
                    glUniform4f(colorLoc, 0.8f, 0.8f, 0.8f, 1.0f);
                    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(ident));
                    glLineWidth(2.0f);
                    glDrawArrays(GL_LINE_LOOP, 0, circleSegments);
                    glDeleteVertexArrays(1, &circleVAO);
                    glDeleteBuffers(1, &circleVBO);
                }
            }
        }
        
        // --- Draw Gizmo Arrows at the End Effector ---
        glm::vec3 ee = positions.back();
        for (int i = 0; i < 3; i++){
            glm::vec3 dir;
            glm::vec3 col;
            if(i == 0) { dir = glm::vec3(1,0,0); col = glm::vec3(1,0,0); }
            else if(i == 1) { dir = glm::vec3(0,1,0); col = glm::vec3(0,1,0); }
            else { dir = glm::vec3(0,0,1); col = glm::vec3(0,0,1); }
            glm::vec3 tip = ee + arrowLength * dir;
            float arrowVerts[6] = { ee.x, ee.y, ee.z, tip.x, tip.y, tip.z };
            GLuint arrowVAO, arrowVBO;
            glGenVertexArrays(1, &arrowVAO);
            glGenBuffers(1, &arrowVBO);
            glBindVertexArray(arrowVAO);
            glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVerts), arrowVerts, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glUniform4f(colorLoc, col.r, col.g, col.b, 1.0f);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(ident));
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, 2);
            glDeleteVertexArrays(1, &arrowVAO);
            glDeleteBuffers(1, &arrowVBO);
        }
        
        // --- Draw Particles ---
		glDisable(GL_DEPTH_TEST);
        for (auto &p : particles) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), p.pos);
            model = glm::scale(model, glm::vec3(0.05f));
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniform4f(colorLoc, 1.0f, 0.0f, 0.0f, particleAlpha);
            glBindVertexArray(cubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        glEnable(GL_DEPTH_TEST);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);
    glDeleteVertexArrays(1, &lineVAO);
    glDeleteBuffers(1, &lineVBO);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}