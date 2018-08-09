#include "vis.hpp"

struct Framebuffer {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> buf;

    void clear(int w_, int h_) {
        if (w != w_ || h != h_) { w = w_; h = h_; }
        buf.assign(w*h * 3, 0);
    }

    void setPixel(int x, int y, const glm::vec3& c) {
        if (x < 0 || w <= x || y < 0 || h <= y) { return; }
        const int i = 3 * (y*w + x);
        buf[i]     = glm::clamp(int(c.r*255), 0, 255);
        buf[i + 1] = glm::clamp(int(c.g*255), 0, 255);
        buf[i + 2] = glm::clamp(int(c.b*255), 0, 255);
    };
};

int main(int argc, char* argv[]) {
    // Parse arguments
    if (argc != 2) {
        std::cerr << "Invalid number of argument(s)" << std::endl;
        return EXIT_FAILURE;
    }

    // Load scene
    Scene scene;
    if (!scene.load(argv[1])) {
        return EXIT_FAILURE;
    }

    // Execute application
    App app;
    if (!app.setup("raster02_vertex_processing")) {
        return EXIT_FAILURE;
    }
    app.run([&](int w, int h, const glm::mat4& viewM) -> App::Buf {
        static Framebuffer fb;
        fb.clear(w, h);

        // Transformation matrix
        const auto modelM = glm::rotate(float(ImGui::GetTime()), glm::vec3(0.f, 1.f, 0.f));
        const auto projM = glm::perspective(glm::radians(30.f), float(fb.w) / fb.h, 0.1f, 10.f);
        const auto transMVP = projM * viewM * modelM;

        // For each triangles in the scene
        scene.foreachTriangles([&](const Scene::Tri& tri) {
            // Model, view, projection transform
            const auto v1 = transMVP * glm::vec4(tri.v1.p, 1);
            const auto v2 = transMVP * glm::vec4(tri.v2.p, 1);
            const auto v3 = transMVP * glm::vec4(tri.v3.p, 1);

            // Perspective division
            const auto v1_ndc = glm::vec3(v1) / v1.w;
            const auto v2_ndc = glm::vec3(v2) / v2.w;
            const auto v3_ndc = glm::vec3(v3) / v3.w;

            // Viewport transform
            const auto viewportTrans = [&](const glm::vec3& p) -> glm::vec2 {
                return glm::vec2(
                    (p.x + 1.f) * .5f * fb.w,
                    (p.y + 1.f) * .5f * fb.h
                );
            };
            const auto v1_w = viewportTrans(v1_ndc);
            const auto v2_w = viewportTrans(v2_ndc);
            const auto v3_w = viewportTrans(v3_ndc);
            
            // Draw pixels
            fb.setPixel(int(v1_w.x), int(v1_w.y), glm::vec3(1.f));
            fb.setPixel(int(v2_w.x), int(v2_w.y), glm::vec3(1.f));
            fb.setPixel(int(v3_w.x), int(v3_w.y), glm::vec3(1.f));
        });

        return { fb.w, fb.h, fb.buf.data() };
    });
    app.shutdown();

    return EXIT_SUCCESS;
}
