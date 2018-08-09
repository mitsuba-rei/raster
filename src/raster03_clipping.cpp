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

    // Clip triangle
    const auto clipTriangle = [&](
        const glm::vec4& v1, const glm::vec4& v2, const glm::vec4& v3,
        const std::function<void(
            const glm::vec4& v1_c,
            const glm::vec4& v2_c,
            const glm::vec4& v3_c)>& processClippedTriangle
    ) {
        // Polygon as a vector of vertices in CCW order
        static std::vector<glm::vec4> poly;
        poly.clear();
        poly.insert(poly.end(), { v1, v2, v3 });

        // Perform clipping
        const glm::vec4 clip_plane_ns[] = {
            glm::vec4( 1, 0, 0, 1),    // w=x
            glm::vec4(-1, 0, 0, 1),    // w=-x
            glm::vec4( 0, 1, 0, 1),    // w=y
            glm::vec4( 0,-1, 0, 1),    // w=-y
            glm::vec4( 0, 0, 1, 1),    // w=z
            glm::vec4( 0, 0,-1, 1),    // w=-z
        };
        for (const auto& clip_plane_n : clip_plane_ns) {
            static std::vector<glm::vec4> outpoly;
            outpoly.clear();
            for (int i = 0; i < int(poly.size()); i++) {
                // Current edge
                const auto& v1 = poly[i];
                const auto& v2 = poly[(i + 1) % poly.size()];

                // Signed distance
                const auto d1 = glm::dot(v1, clip_plane_n);
                const auto d2 = glm::dot(v2, clip_plane_n);

                // Calculate intersection between a segment and a clip plane
                const auto intersect = [&]() -> glm::vec4 {
                    const auto a = d1 / (d1 - d2);
                    return (1.f - a)*v1 + a * v2;
                };
                if (d1 > 0) {
                    if (d2 > 0) {
                        // Both inside
                        outpoly.push_back(v2);
                    }
                    else {
                        // p1: indide, p2: outside
                        outpoly.push_back(intersect());
                    }
                }
                else if (d2 > 0) {
                    // p1: outside, p2: inside
                    outpoly.push_back(intersect());
                    outpoly.push_back(v2);
                }
            }

            poly = outpoly;
        }

        // Triangulate the polygon
        if (poly.empty()) {
            return;
        }
        const auto& vt1 = poly[0];
        for (int i = 1; i < int(poly.size()) - 1; i++) {
            const auto& vt2 = poly[i];
            const auto& vt3 = poly[(i + 1) % poly.size()];
            processClippedTriangle(vt1, vt2, vt3);
        }
    };

    // Rasterize
    const auto rasterLine = [&](Framebuffer& fb, const glm::vec2& p1, const glm::vec2& p2) {
        int x1 = int(p1.x);
        int y1 = int(p1.y);
        int x2 = int(p2.x);
        int y2 = int(p2.y);
        bool trans = false;
        if (abs(x2 - x1) < abs(y2 - y1)) {
            std::swap(x1, y1);
            std::swap(x2, y2);
            trans = true;
        }
        if (x1 > x2) {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        const int dx = x2 - x1;
        const int dy = y2 - y1;
        const int delta = abs(dy) * 2;
        const int yd = dy > 0 ? 1 : -1;
        int error = 0;
        int y = y1;
        for (int x = x1; x <= x2; x++) {
            fb.setPixel(trans ? y : x, trans ? x : y, glm::vec3(1));
            error += delta;
            if (error > dx) {
                y += yd;
                error -= dx * 2;
            }
        }
    };

    // Execute application
    App app;
    if (!app.setup("raster03_clipping")) {
        return EXIT_FAILURE;
    }
    app.run([&](int w, int h, const glm::mat4& viewM) -> App::Buf {
        // Animation
        static bool animate = true;
        ImGui::Checkbox("Enable animation", &animate);
        ImGui::Separator();

        // Camera parameters
        static float fov = 30.f;
        static float znear = 0.1f;
        static float zfar = 10.f;
        ImGui::DragFloat("fov", &fov, 0.1f, 0.01f, 180.f);
        ImGui::DragFloat("near", &znear, 0.01f, 0.01f, 10.f);
        ImGui::DragFloat("far", &zfar, 0.01f, 1.f, 1000.f);
        ImGui::Separator();

        // Framebuffer
        static Framebuffer fb;
        fb.clear(w, h);

        // Transformation matrix
        const auto modelM = animate ? glm::rotate(ImGui::GetTime(), glm::vec3(0, 1, 0)) : glm::mat4(1.f);
        const auto projM = glm::perspective(glm::radians(fov), float(fb.w) / fb.h, znear, zfar);
        const auto transMVP = projM * viewM * modelM;

        // For each triangles in the scene
        scene.foreachTriangles([&](const Scene::Tri& tri) {
            // Model, view, projection transform
            const auto v1 = transMVP * glm::vec4(tri.v1.p, 1);
            const auto v2 = transMVP * glm::vec4(tri.v2.p, 1);
            const auto v3 = transMVP * glm::vec4(tri.v3.p, 1);

            // Clip triangle and raster
            clipTriangle(v1, v2, v3, [&](const glm::vec4& v1_c, const glm::vec4& v2_c, const glm::vec4& v3_c) {
                // Perspective division
                const auto v1_ndc = glm::vec3(v1_c) / v1_c.w;
                const auto v2_ndc = glm::vec3(v2_c) / v2_c.w;
                const auto v3_ndc = glm::vec3(v3_c) / v3_c.w;

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

                // Raster lines
                rasterLine(fb, v1_w, v2_w);
                rasterLine(fb, v2_w, v3_w);
                rasterLine(fb, v3_w, v1_w);
            });
        });

        return { fb.w, fb.h, fb.buf.data() };
    });
    app.shutdown();

    return EXIT_SUCCESS;
}
