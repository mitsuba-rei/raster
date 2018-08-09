#include "vis.hpp"

#define UNREACHABLE() assert(0)
constexpr float Inf = 1e+8f;
constexpr float Eps = 1e-4f;

struct Framebuffer {
    int w = 0;
    int h = 0;
    std::vector<unsigned char> buf;     // Color buffer
    std::vector<float> zbuf;            // Depth buffer

    void clear(int w_, int h_) {
        if (w != w_ || h != h_) { w = w_; h = h_; }
        buf.assign(w*h * 3, 0);
        zbuf.assign(w*h * 3, Inf);
    }

    void setPixel(int x, int y, const glm::vec3& c) {
        if (x < 0 || w <= x || y < 0 || h <= y) { return; }
        const int i = 3 * (y*w + x);
        buf[i]     = glm::clamp(int(c.r * 255), 0, 255);
        buf[i + 1] = glm::clamp(int(c.g * 255), 0, 255);
        buf[i + 2] = glm::clamp(int(c.b * 255), 0, 255);
    };
};

struct VaryingVert {
    glm::vec4 p;
    glm::vec3 n;
};
using VertexShaderFunc = std::function<VaryingVert(const Scene::Vert& v)>;
using FragmentShaderFunc = std::function<glm::vec3(const glm::vec3& n, const glm::vec3& p_ndc)>;

void rasterize(
    const Scene& scene,
    Framebuffer& fb,
    bool wireframe,
    bool cullbackface,
    const VertexShaderFunc& vertexShader,
    const FragmentShaderFunc& fragmentShader
) {
    // Viewport transform
    const auto viewportTrans = [&](const glm::vec3& p) -> glm::vec2 {
        return glm::vec2(
            (p.x + 1.f) * .5f * fb.w,
            (p.y + 1.f) * .5f * fb.h
        );
    };

    // Rasterize a line segment
    const auto rasterLine = [&](const glm::vec2& p1, const glm::vec2& p2) {
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
        
    // Rasterize a triangle
    const auto rasterTriangle = [&](const VaryingVert& v1, const VaryingVert& v2, const VaryingVert& v3) {
        // Clip space -> NDC -> Screen space
        const auto p1_ndc = glm::vec3(v1.p) / v1.p.w;
        const auto p2_ndc = glm::vec3(v2.p) / v2.p.w;
        const auto p3_ndc = glm::vec3(v3.p) / v3.p.w;
        const auto p1 = viewportTrans(p1_ndc);
        const auto p2 = viewportTrans(p2_ndc);
        const auto p3 = viewportTrans(p3_ndc);

        // Wireframe?
        if (wireframe) {
            rasterLine(p1, p2);
            rasterLine(p2, p3);
            rasterLine(p3, p1);
            return;
        }

        // Bounding box in screen coordinates
        glm::vec2 min( Inf);
        glm::vec2 max(-Inf);
        const auto merge = [&](const glm::vec2& p) {
            min = glm::min(min, p);
            max = glm::max(max, p);
        };
        merge(p1);
        merge(p2);
        merge(p3);
        min = glm::max(min, glm::vec2(0));
        max = glm::min(max, glm::vec2(fb.w-1, fb.h-1));

        // Edge function (CCW)
        const auto edgeFunc = [](const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
            const auto d1 = b - a;
            const auto d2 = c - a;
            return d1.x*d2.y - d1.y*d2.x;
        };

        // Check inside/outside tests for each pixel
        const auto denom = edgeFunc(p1, p2, p3);
        const bool back = denom < 0;
        if (back && cullbackface) {
            return;
        }
        for (int y = int(min.y); y <= int(max.y); y++) {
            for (int x = int(min.x); x <= int(max.x); x++) {
                const auto p = glm::vec2(x, y) + 0.5f;
                auto b1 = edgeFunc(p2, p3, p);
                auto b2 = edgeFunc(p3, p1, p);
                auto b3 = edgeFunc(p1, p2, p);
                const bool inside = (b1>0 && b2>0 && b3>0) || (b1<0 && b2<0 && b3<0);
                if (!inside) {
                    continue;
                }
                b1 /= denom;
                b2 /= denom;
                b3 /= denom;
                const auto p_ndc = b1 * p1_ndc + b2 * p2_ndc + b3 * p3_ndc;
                if (fb.zbuf[y*fb.w+x] < p_ndc.z) {
                    continue;
                }
                fb.zbuf[y*fb.w + x] = p_ndc.z;
                const auto n = glm::normalize(b1/v1.p.w*v1.n + b2/v2.p.w*v2.n + b3/v3.p.w*v3.n);
                fb.setPixel(x, y, fragmentShader(back ? -n : n, p_ndc));
            }
        }
    };

    // Clip triangle
    const auto clipTriangle = [&](
        const VaryingVert& v1,
        const VaryingVert& v2,
        const VaryingVert& v3,
        const std::function<void (
            const VaryingVert& v1_clipped,
            const VaryingVert& v2_clipped,
            const VaryingVert& v3_clipped)>& processClippedTriangle
    ) {
        // Polygon as a vector of vertices in CCW order
        static std::vector<VaryingVert> poly;
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
            static std::vector<VaryingVert> outpoly;
            outpoly.clear();
            for (int i = 0; i < int(poly.size()); i++) {
                // Current edge
                const auto& v1 = poly[i];
                const auto& v2 = poly[(i+1)%poly.size()];

                // Signed distance
                const auto d1 = glm::dot(v1.p, clip_plane_n);
                const auto d2 = glm::dot(v2.p, clip_plane_n);

                // Calculate intersection between a segment and a clip plane
                const auto intersect = [&]() -> VaryingVert {
                    const auto a = d1 / (d1 - d2);
                    return {
                        (1.f-a)*v1.p + a*v2.p,
                        glm::normalize((1.f-a)*v1.n + a*v2.n)
                    };
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
            const auto& vt3 = poly[(i+1)%poly.size()];
            processClippedTriangle(vt1, vt2, vt3);
        }
    };

    // For each triangles in the scene
    scene.foreachTriangles([&](const Scene::Tri& tri) {
        // Transform vertices
        const auto v1 = vertexShader(tri.v1);
        const auto v2 = vertexShader(tri.v2);
        const auto v3 = vertexShader(tri.v3);

        // Clip triangle and raster
        clipTriangle(v1, v2, v3, [&](const VaryingVert& v1_c, const VaryingVert& v2_c, const VaryingVert& v3_c) {
            rasterTriangle(v1_c, v2_c, v3_c);
        });
    });
}

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
    if (!app.setup("raster05_shadow")) {
        return EXIT_FAILURE;
    }
    app.run([&](int w, int h, const glm::mat4& viewM) -> App::Buf {
        // Raster mode
        enum class RasterMode {
            Shaded,
            Normal,
            Wireframe,
        };
        const auto mode = [&]() {
            static int mode = 0;
            ImGui::RadioButton("Shaded", &mode, 0); ImGui::SameLine();
            ImGui::RadioButton("Normal", &mode, 1); ImGui::SameLine();
            ImGui::RadioButton("Wireframe", &mode, 2);
            ImGui::Separator();
            return RasterMode(mode);
        }();

        // Animation
        static bool animate = true;
        ImGui::Checkbox("Enable animation", &animate);
        ImGui::Separator();

        // Backface culling
        static bool cullbackface = true;
        static bool cullbackface_shadow = false;
        ImGui::Checkbox("Enable backface culling", &cullbackface);
        ImGui::Checkbox("Enable backface culling (shadow)", &cullbackface_shadow);
        ImGui::Separator();

        // Camera parameters
        static float fov = 30.f;
        static float znear = 0.1f;
        static float zfar = 10.f;
        ImGui::DragFloat("fov", &fov, 0.1f, 0.01f, 180.f);
        ImGui::DragFloat("near", &znear, 0.01f, 0.01f, 10.f);
        ImGui::DragFloat("far", &zfar, 0.01f, 1.f, 1000.f);
        ImGui::Separator();

        // Shadow map resolution
        static int shadowmapSize = 1000;
        ImGui::DragInt("Shadow map size", &shadowmapSize, 10, 100, 5000);
        ImGui::Separator();

        // Framebuffers
        static Framebuffer fb;
        static Framebuffer fb_shadow;

        // Transformation matrix
        const auto modelM = animate ? glm::rotate(ImGui::GetTime(), glm::vec3(0,1,0)) : glm::mat4(1.f);
        const auto transN = glm::mat3(glm::transpose(glm::inverse(modelM)));

        // Direction of light
        const auto lightdir = glm::normalize(glm::vec3(0.5, 0.8, 1));

        // Pass 1: Shadow
        glm::mat4 transMVP_shadow;
        if (mode == RasterMode::Shaded) {
            fb_shadow.clear(shadowmapSize, shadowmapSize);
            const auto s = 3.f;
            const auto projM = glm::ortho(-s, s, -s, s, -5.f, 5.f);
            const auto viewM = glm::lookAt(lightdir, glm::vec3(), glm::vec3(0,1,0));
            transMVP_shadow = projM * viewM * modelM;
            rasterize(scene, fb_shadow, false, cullbackface_shadow,
                // Vertex shader
                [&](const Scene::Vert& v) -> VaryingVert {
                    return { transMVP_shadow * glm::vec4(v.p, 1), transN * v.n };
                },
                // Fragment shader
                [&](const glm::vec3& n, const glm::vec3& p_ndc) -> glm::vec3 {
                    return glm::vec3((p_ndc.z + 1.f) * .5f);
                });
        }

        // Pass 2: Color
        {
            fb.clear(w, h);
            const auto projM = glm::perspective(glm::radians(fov), float(fb.w) / fb.h, znear, zfar);
            const auto transMVP = projM * viewM * modelM;
            const auto transShadow = transMVP_shadow * glm::inverse(transMVP);
            rasterize(scene, fb, mode == RasterMode::Wireframe, cullbackface,
                // Vertex shader
                [&](const Scene::Vert& v) -> VaryingVert {
                    return { transMVP * glm::vec4(v.p, 1), transN * v.n };
                },
                // Fragment shader
                [&](const glm::vec3& n, const glm::vec3& p_ndc) -> glm::vec3 {
                    if (mode == RasterMode::Normal) {
                        return glm::abs(n);
                    }
                    else if (mode == RasterMode::Shaded) {
                        // Shadow mapping with percentage closer filtering (PCF)
                        auto p = transShadow * glm::vec4(p_ndc, 1);
                        p /= p.w;
                        const auto costheta = glm::dot(n, lightdir);
                        const auto bias = glm::clamp(0.01f*(1.f - costheta), 0.001f, 0.01f);
                        float vis = 0.f;
                        const int px = int((p.x + 1.f) * .5f * fb_shadow.w);
                        const int py = int((p.y + 1.f) * .5f * fb_shadow.h);
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                const int x = px + dx;
                                const int y = py + dy;
                                if (0 <= x && x<fb_shadow.w && 0 <= y && y<fb_shadow.h) {
                                    const auto z = fb_shadow.zbuf[y*fb_shadow.w + x];
                                    vis += z + bias < p.z ? 0.f : 1.f;
                                }
                            }
                        }
                        vis /= 9.f;
                        vis = .5f + vis * .5f;
                        return glm::vec3(0.2f + glm::max(0.f, vis*0.8f*costheta));
                    }
                    UNREACHABLE();
                    return {};
                });
        }

        // Select framebuffer output
        static int selectedMap = 0;
        ImGui::RadioButton("Color map", &selectedMap, 0); ImGui::SameLine();
        ImGui::RadioButton("Shadow map", &selectedMap, 1);
        if (selectedMap == 0) {
            return { fb.w, fb.h, fb.buf.data() };
        }
        else if (selectedMap == 1) {
            return { fb_shadow.w, fb_shadow.h, fb_shadow.buf.data() };
        }
        UNREACHABLE();
        return {};
    });
    app.shutdown();

    return EXIT_SUCCESS;
}
