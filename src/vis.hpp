#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <tuple>
#include <algorithm>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define IMAGE_VISUALIZER_USE_PBO 1
#if IMAGE_VISUALIZER_USE_PBO
constexpr int NumPBOs = 2;
#endif

struct Scene {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    bool load(const std::string& path) {
        std::string warn;
        std::string err;
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
            std::cerr << warn << std::endl;
            std::cerr << err << std::endl;
            return false;
        }
        return true;
    }

    struct Vert {
        glm::vec3 p;
        glm::vec3 n;
    };
    struct Tri {
        Vert v1;
        Vert v2;
        Vert v3;
    };
    void foreachTriangles(const std::function<void(const Tri& tri)>& iterfunc) const {
        for (const auto& shape : shapes) {
            for (int fi = 0; fi < int(shape.mesh.num_face_vertices.size()); fi++) {
                const auto v = [&](int index) -> Vert {
                    const auto i = shape.mesh.indices[index];
                    return {
                        glm::vec3(
                            attrib.vertices[3*i.vertex_index+0],
                            attrib.vertices[3*i.vertex_index+1],
                            attrib.vertices[3*i.vertex_index+2]),
                        glm::normalize(glm::vec3(
                            attrib.normals[3*i.normal_index+0],
                            attrib.normals[3*i.normal_index+1],
                            attrib.normals[3*i.normal_index+2]))
                    };
                };
                iterfunc({ v(3*fi+0), v(3*fi+1), v(3*fi+2) });
            }
        }
    }
};

// Image visualizer with OpenGL
class ImageVisualizer {
private:
    GLuint pipeline_;        // Pipeline object
    GLuint programV_;        // Vertex shader
    GLuint programF_;        // Fragment shader
    GLuint texture_;         // Texture
    GLuint bufferP_;         // Vertex buffer for positions
    GLuint bufferI_;         // Index buffer
    GLuint vertexArray_;     // Vertex array object
    #if IMAGE_VISUALIZER_USE_PBO
    GLuint pbo_[NumPBOs];
    int currPboIndex_ = 0;
    #endif

public:
    bool init() {
        #pragma region Programs
        #ifdef __APPLE__
        const char* vscode = R"x(
            #version 410 core
            layout (location = 0) in vec2 position;
            out gl_PerVertex {
                vec4 gl_Position;
            };
            out vec2 uv;
            void main() {
                uv = (position + 1) * .5;
                gl_Position = vec4(position, 0, 1);
            }
        )x";
        const char* fscode = R"x(
            #version 410 core
            /*layout (binding = 0)*/ uniform sampler2D tex;
            in vec2 uv;
            out vec4 color;
            void main() {
                color = texture(tex, uv);
            }
        )x";
        #else
        const char* vscode = R"x(
            #version 430 core
            layout (location = 0) in vec2 position;
            out gl_PerVertex {
                vec4 gl_Position;
            };
            out vec2 uv;
            void main() {
                uv = (position + 1) * .5;
                gl_Position = vec4(position, 0, 1);
            }
        )x";
        const char* fscode = R"x(
            #version 430 core
            layout (binding = 0) uniform sampler2D tex;
            in vec2 uv;
            out vec4 color;
            void main() {
                color = texture(tex, uv);
            }
        )x";
        #endif
        const auto createProgram = [](GLenum shaderType, const std::string& code) -> std::optional<GLuint> {
            GLuint program = glCreateProgram();
            GLuint shader = glCreateShader(shaderType);
            const auto* codeptr = code.c_str();
            glShaderSource(shader, 1, &codeptr, nullptr);
            glCompileShader(shader);
            GLint ret;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ret);
            if (!ret) {
                int length;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
                std::vector<char> v(length);
                glGetShaderInfoLog(shader, length, nullptr, &v[0]);
                glDeleteShader(shader);
                std::cerr << std::string(&v[0]) << std::endl;
                return {};
            }
            glAttachShader(program, shader);
            glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
            glDeleteShader(shader);
            glLinkProgram(program);
            glGetProgramiv(program, GL_LINK_STATUS, &ret);
            if (!ret) {
                GLint length;
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
                std::vector<char> v(length);
                glGetProgramInfoLog(program, length, nullptr, &v[0]);
                std::cerr << std::string(&v[0]) << std::endl;
                return false;
            }
            return program;
        };
        {
            const auto vp = createProgram(GL_VERTEX_SHADER, vscode);
            const auto fp = createProgram(GL_FRAGMENT_SHADER, fscode);
            if (!vp || !fp) {
                return false;
            }
            programV_ = *vp;
            programF_ = *fp;
        }
        glGenProgramPipelines(1, &pipeline_);
        glUseProgramStages(pipeline_, GL_VERTEX_SHADER_BIT, programV_);
        glUseProgramStages(pipeline_, GL_FRAGMENT_SHADER_BIT, programF_);
        #pragma endregion

        // --------------------------------------------------------------------------------

        #pragma region Vertex
        const std::vector<float> vs{ -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f };
        const std::vector<unsigned int> is{ 0, 1, 2, 0, 2, 3 };

        glGenBuffers(1, &bufferP_);
        glBindBuffer(GL_ARRAY_BUFFER, bufferP_);
        glBufferData(GL_ARRAY_BUFFER, vs.size() * sizeof(float), vs.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glGenBuffers(1, &bufferI_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferI_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, is.size() * sizeof(unsigned int), is.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glGenVertexArrays(1, &vertexArray_);
        glBindVertexArray(vertexArray_);
        glBindBuffer(GL_ARRAY_BUFFER, bufferP_);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        #pragma endregion

        // --------------------------------------------------------------------------------

        #pragma region Texture
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        #pragma endregion

        // --------------------------------------------------------------------------------

        #if IMAGE_VISUALIZER_USE_PBO
        glGenBuffers(NumPBOs, pbo_);
        for (int i = 0; i < NumPBOs; i++) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[i]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, 1, nullptr, GL_STREAM_DRAW);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        #endif

        // --------------------------------------------------------------------------------

        return true;
    }

    void update(int w, int h, const unsigned char* data) {
        glBindTexture(GL_TEXTURE_2D, texture_);
        // Get current size of the texture
        int w_, h_;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w_);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h_);
        if (w != w_ || h != h_) {
            // Change of the texture is detected, reallocate
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            #if IMAGE_VISUALIZER_USE_PBO
            for (int i = 0; i < NumPBOs; i++) {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[i]);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h*4, data, GL_STREAM_DRAW);
            }
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            #endif
        }
        else {
            // Update the texture
            #if IMAGE_VISUALIZER_USE_PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[currPboIndex_]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            const int nextPboIndex = (currPboIndex_ + 1) % NumPBOs;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[nextPboIndex]);
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, w*h*4, data);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            currPboIndex_ = nextPboIndex;
            #else
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data);
            #endif
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void render() {
        glBindProgramPipeline(pipeline_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glBindVertexArray(vertexArray_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferI_);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindProgramPipeline(0);
    }
};

// Main application
class App {
private:
    GLFWwindow* window_;
    ImageVisualizer vis_;

public:
    bool setup(const std::string& windowName) {
        // Init GLFW
        if (!glfwInit()) {
            return false;
        }

        // Craete GLFW window
        window_ = [&]() -> GLFWwindow* {
            // GLFW window
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            #ifdef __APPLE__
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
            #else
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            #endif
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            #ifdef _DEBUG
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
            #endif
            GLFWwindow* window = glfwCreateWindow(1280, 720, windowName.c_str(), nullptr, nullptr);
            if (!window) { return nullptr; }
            glfwMakeContextCurrent(window);
            glfwSwapInterval(0);
            gl3wInit();
            // ImGui context
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGui_ImplGlfw_InitForOpenGL(window, true);
            #ifdef __APPLE__
            ImGui_ImplOpenGL3_Init("#version 410 core");
            #else
            ImGui_ImplOpenGL3_Init();
            #endif
            ImGui::StyleColorsDark();
            return window;
        }();
        if (!window_) {
            glfwTerminate();
            return false;
        }

        // Image visualizer
        if (!vis_.init()) {
            return false;
        }

        // Row alignment (default: 4)
        // https://www.khronos.org/opengl/wiki/Common_Mistakes#Texture_upload_and_pixel_reads
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        #ifdef _DEBUG
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback([](
            GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
            const GLchar* message, void* userParam) -> void {
            fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
                (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
        }, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, true);
        #endif

        return true;
    }

    void shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    struct Buf {
        int w;
        int h;
        unsigned char* data;
    };
    void run(const std::function<Buf(int w, int h, const glm::mat4& viewM)>& updateFunc) {
        while (!glfwWindowShouldClose(window_)) {
            #pragma region Setup new frame
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            #pragma endregion

            // --------------------------------------------------------------------------------

            #pragma region Update camera
            const static auto init_eye = glm::vec3(0,1.5,4.5);
            const static auto init_center = glm::vec3(0,.5,0);
            const static auto init_d = glm::normalize(init_center - init_eye);
            const auto viewM = [&]() {
                const glm::vec3 up(0, 1, 0);

                // Camera rotation
                const auto forward = [&]() -> glm::vec3 {
                    static float pitch = glm::degrees(asin(init_d.y));
                    static float yaw = glm::degrees(atan2(init_d.z, init_d.x));
                    static auto prevMousePos = ImGui::GetMousePos();
                    const auto mousePos = ImGui::GetMousePos();
                    const bool rotating = ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_RIGHT);
                    if (rotating) {
                        int w, h;
                        glfwGetFramebufferSize(window_, &w, &h);
                        const float sensitivity = 0.1f;
                        const float dx = float(prevMousePos.x - mousePos.x) * sensitivity;
                        const float dy = float(prevMousePos.y - mousePos.y) * sensitivity;
                        yaw += dx;
                        pitch = glm::clamp(pitch - dy, -89.f, 89.f);
                    }
                    prevMousePos = mousePos;
                    return glm::vec3(
                        cos(glm::radians(pitch)) * cos(glm::radians(yaw)),
                        sin(glm::radians(pitch)),
                        cos(glm::radians(pitch)) * sin(glm::radians(yaw)));
                }();

                // Camera position
                const auto p = [&]() -> glm::vec3 {
                    static auto p = init_eye;
                    const auto w = -forward;
                    const auto u = glm::normalize(glm::cross(up, w));
                    const auto v = glm::cross(w, u);
                    const float factor = ImGui::GetIO().KeyShift ? 10.0f : 1.f;
                    const float speed = ImGui::GetIO().DeltaTime * factor;
                    if (ImGui::IsKeyDown('W')) { p += forward * speed; }
                    if (ImGui::IsKeyDown('S')) { p -= forward * speed; }
                    if (ImGui::IsKeyDown('A')) { p -= u * speed; }
                    if (ImGui::IsKeyDown('D')) { p += u * speed; }
                    return p;
                }();

                return glm::lookAt(p, p+forward, up);
            }();
            #pragma endregion

            // --------------------------------------------------------------------------------

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(350, 350), ImGuiCond_Once);

            // General information
            ImGui::Begin("Information / Control");
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            ImGui::Text("Framebuffer size: (%d, %d)", display_w, display_h);
            ImGui::Text("Framebuffer streaming mode: %s", IMAGE_VISUALIZER_USE_PBO ? "PBO" : "Normal");
            #if IMAGE_VISUALIZER_USE_PBO
            ImGui::Text("Number of PBOs: %d", NumPBOs);
            #endif
            ImGui::Separator();

            // User-defined update
            {
                const auto buf = updateFunc(display_w, display_h, viewM);
                vis_.update(buf.w, buf.h, buf.data);
            }

            ImGui::End();

            // --------------------------------------------------------------------------------

            #pragma region Render frame
            ImGui::Render();
            glfwMakeContextCurrent(window_);
            glViewport(0, 0, display_w, display_h);
            glClearColor(.45f, .55f, .6f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);
            vis_.render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwMakeContextCurrent(window_);
            glfwSwapBuffers(window_);
            #pragma endregion
        }
    }
};