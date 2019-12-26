#include "vis.hpp"

#define UNREACHABLE() assert(0)
constexpr float Inf = 1e+8f;
constexpr float Eps = 1e-4f;

#pragma GCC target("avx")
#pragma clang attribute push (__attribute__((target("avx"))), apply_to=function)
#include <x86intrin.h>
static inline void clear(unsigned char *buf, float *zbuf, size_t count);
#pragma clang attribute pop

static inline void clear(unsigned char *buf, float *zbuf, const size_t count) {
    static const __m256 zero = _mm256_set1_ps(0);
    static const __m256 inf = _mm256_set1_ps(Inf);
    float *pb = (float *)buf;
    for (size_t i = 0; i < count; i += 8) {
        _mm256_store_ps(pb + i, zero);
        _mm256_store_ps(zbuf + i, inf);
    }
}

struct Framebuffer {
    int w = 0;
    int h = 0;

    uintptr_t buforg = 0;  // Color buffer
    uintptr_t zbuforg = 0; // Depth buffer

    unsigned char *buf = nullptr;     // Color buffer (aligned)
    float *zbuf = nullptr;            // Depth buffer (aligned)

    void clear(int w_, int h_) {
        if (w != w_ || h != h_) {
            w = w_;
            h = h_;
            ::free((void *)buforg);
            ::free((void *)zbuforg);
            buforg = (uintptr_t)::malloc(w*h * 4 + 31);
            zbuforg = (uintptr_t)::malloc(w*h * 4 + 31);
            buf = (unsigned char *)((buforg + 31) & ~31ULL);
            zbuf = (float *)((zbuforg + 31) & ~31ULL);
        }
#if 0
        ::memset(buf, 0, w*h * 4);
        //zbuf.assign(w*h, Inf);
        float *p = zbuf;
        for (size_t i = 0; i < w*h; i++) {
            *p++ = Inf;
        }
#else
        ::clear(buf, zbuf, w*h);
#endif
    }

    void setPixel(int x, int y, const glm::vec3& c) {
        if (x < 0 || w <= x || y < 0 || h <= y) { return; }
        const size_t i = (w * y + x) * 4;
        buf[i + 0] = glm::clamp(int(c.r * 255), 0, 255);
        buf[i + 1] = glm::clamp(int(c.g * 255), 0, 255);
        buf[i + 2] = glm::clamp(int(c.b * 255), 0, 255);
        buf[i + 3] = 255;
    };
};

struct VaryingVert {
    glm::vec4 p;
    glm::vec3 n;
};
using VertexShaderFunc = std::function<VaryingVert(const Scene::Vert& v)>;
using FragmentShaderFunc = std::function<glm::vec3(const glm::vec3& n, const glm::vec3& p_ndc)>;

#pragma GCC target("avx2")
#pragma clang attribute push (__attribute__((target("avx2, fma"))), apply_to=function)
#include <x86intrin.h>
void rasterize(
    const Scene& scene,
    Framebuffer& fb,
    bool wireframe,
    bool cullbackface,
    const VertexShaderFunc& vertexShader,
    const FragmentShaderFunc& fragmentShader
);
#pragma clang attribute pop

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
        // alignment
        static const int ALIGN = 8;
        int minX = int(min.x) & ~(ALIGN - 1);
        int maxX = (int(max.x) + 1 + (ALIGN - 1)) & ~(ALIGN - 1);

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
        #if 0
        const float p1_ndc_z = p1_ndc.z;
        const float p2_ndc_z = p2_ndc.z;
        const float p3_ndc_z = p3_ndc.z;
        const float denominv = 1.0f/denom;
        const float v1_n_x = v1.n.x / v1.p.w;
        const float v1_n_y = v1.n.y / v1.p.w;
        const float v1_n_z = v1.n.z / v1.p.w;
        const float v2_n_x = v2.n.x / v2.p.w;
        const float v2_n_y = v2.n.y / v2.p.w;
        const float v2_n_z = v2.n.z / v2.p.w;
        const float v3_n_x = v3.n.x / v3.p.w;
        const float v3_n_y = v3.n.y / v3.p.w;
        const float v3_n_z = v3.n.z / v3.p.w;

        const auto d1_1_x = p3.x - p2.x;
        const auto d1_1_y = p3.y - p2.y;

        const auto d2_1_x = p1.x - p3.x;
        const auto d2_1_y = p1.y - p3.y;

        const auto d3_1_x = p2.x - p1.x;
        const auto d3_1_y = p2.y - p1.y;

        const float d1 = -d1_1_y;
        const float d2 = -d2_1_y;
        const float d3 = -d3_1_y;

        for (int y = int(min.y); y <= int(max.y); y++) {
            const auto p_y = y + 0.5f;

            const auto d1_2_y = p_y - p2.y;
            const auto a1_1 = d1_1_x*d1_2_y + d1_1_y*p2.x;

            const auto d2_2_y = p_y - p3.y;
            const auto a2_1 = d2_1_x*d2_2_y + d2_1_y*p3.x;

            const auto d3_2_y = p_y - p1.y;
            const auto a3_1 = d3_1_x*d3_2_y + d3_1_y*p1.x;

            const float a1 = a1_1;
            const float a2 = a2_1;
            const float a3 = a3_1;

            for (int x = int(min.x); x <= int(max.x); x++) {
                const auto p_x = x + 0.5f;
                auto b1 = d1 * p_x + a1;
                auto b2 = d2 * p_x + a2;
                auto b3 = d3 * p_x + a3;
                const bool inside = (b1>0 && b2>0 && b3>0) || (b1<0 && b2<0 && b3<0);
                if (!inside) {
                    continue;
                }
                b1 *= denominv;
                b2 *= denominv;
                b3 *= denominv;
                const auto p_ndc_z = b1 * p1_ndc_z + b2 * p2_ndc_z + b3 * p3_ndc_z;
                const bool depthGE = (fb.zbuf[y*fb.w + x] >= p_ndc_z);
                if (!inside || !depthGE) {
                    continue;
                } else {
                    fb.zbuf[y*fb.w + x] = p_ndc_z;
                }
                const auto n_x = b1 * v1_n_x + b2 * v2_n_x + b3 * v3_n_x;
                const auto n_y = b1 * v1_n_y + b2 * v2_n_y + b3 * v3_n_y;
                const auto n_z = b1 * v1_n_z + b2 * v2_n_z + b3 * v3_n_z;
                auto n = n_x*n_x + n_y*n_y + n_z*n_z;
                n = 1.0f / sqrt(n);
                const auto c = glm::abs(glm::vec3{n_x*n, n_y*n, n_z*n});
                if (inside && depthGE) {
                    fb.setPixel(x, y, c);
                }
            }
        }
        #else
        static const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000
        static const __m256 zero = _mm256_set1_ps(0.0f);
        static const __m256 _255 = _mm256_set1_ps(255.0f);
        const __m256 p1_ndc_z = _mm256_set1_ps(p1_ndc.z);
        const __m256 p2_ndc_z = _mm256_set1_ps(p2_ndc.z);
        const __m256 p3_ndc_z = _mm256_set1_ps(p3_ndc.z);
        const __m256 denominv = _mm256_set1_ps(1.0f/denom);
        const __m256 v1_n_x = _mm256_set1_ps(v1.n.x / v1.p.w);
        const __m256 v1_n_y = _mm256_set1_ps(v1.n.y / v1.p.w);
        const __m256 v1_n_z = _mm256_set1_ps(v1.n.z / v1.p.w);
        const __m256 v2_n_x = _mm256_set1_ps(v2.n.x / v2.p.w);
        const __m256 v2_n_y = _mm256_set1_ps(v2.n.y / v2.p.w);
        const __m256 v2_n_z = _mm256_set1_ps(v2.n.z / v2.p.w);
        const __m256 v3_n_x = _mm256_set1_ps(v3.n.x / v3.p.w);
        const __m256 v3_n_y = _mm256_set1_ps(v3.n.y / v3.p.w);
        const __m256 v3_n_z = _mm256_set1_ps(v3.n.z / v3.p.w);

        const auto d1_1_x = p3.x - p2.x;
        const auto d1_1_y = p3.y - p2.y;

        const auto d2_1_x = p1.x - p3.x;
        const auto d2_1_y = p1.y - p3.y;

        const auto d3_1_x = p2.x - p1.x;
        const auto d3_1_y = p2.y - p1.y;

        const __m256 d1 = _mm256_set1_ps(-d1_1_y);
        const __m256 d2 = _mm256_set1_ps(-d2_1_y);
        const __m256 d3 = _mm256_set1_ps(-d3_1_y);

        for (int y = int(min.y); y <= int(max.y); y++) {
            const auto p_y = y + 0.5f;

            const auto d1_2_y = p_y - p2.y;
            const auto a1_1 = d1_1_x*d1_2_y + d1_1_y*p2.x;

            const auto d2_2_y = p_y - p3.y;
            const auto a2_1 = d2_1_x*d2_2_y + d2_1_y*p3.x;

            const auto d3_2_y = p_y - p1.y;
            const auto a3_1 = d3_1_x*d3_2_y + d3_1_y*p1.x;

            const __m256 a1 = _mm256_set1_ps(a1_1);
            const __m256 a2 = _mm256_set1_ps(a2_1);
            const __m256 a3 = _mm256_set1_ps(a3_1);

            for (int x = minX; x < maxX; x += 8) {
                //const auto p_x = x + 0.5f;
                const __m256 p_x = _mm256_set_ps(
                    (x + 7) + 0.5f,
                    (x + 6) + 0.5f,
                    (x + 5) + 0.5f,
                    (x + 4) + 0.5f,
                    (x + 3) + 0.5f,
                    (x + 2) + 0.5f,
                    (x + 1) + 0.5f,
                    (x + 0) + 0.5f
                );
                //auto b1 = d1 * p_x + a1;
                //auto b2 = d2 * p_x + a2;
                //auto b3 = d3 * p_x + a3;
                __m256 b1 = _mm256_fmadd_ps(d1, p_x, a1);
                __m256 b2 = _mm256_fmadd_ps(d2, p_x, a2);
                __m256 b3 = _mm256_fmadd_ps(d3, p_x, a3);
                //const bool inside = (b1>0 && b2>0 && b3>0) || (b1<0 && b2<0 && b3<0);
                const __m256 gt1 = _mm256_cmp_ps(b1, zero, _CMP_GT_OQ);
                const __m256 gt2 = _mm256_cmp_ps(b2, zero, _CMP_GT_OQ);
                const __m256 gt3 = _mm256_cmp_ps(b3, zero, _CMP_GT_OQ);
                const __m256 lt1 = _mm256_cmp_ps(b1, zero, _CMP_LT_OQ);
                const __m256 lt2 = _mm256_cmp_ps(b2, zero, _CMP_LT_OQ);
                const __m256 lt3 = _mm256_cmp_ps(b3, zero, _CMP_LT_OQ);
                const __m256 gt123 = _mm256_and_ps(_mm256_and_ps(gt1, gt2), gt3);
                const __m256 lt123 = _mm256_and_ps(_mm256_and_ps(lt1, lt2), lt3);
                const __m256 inside = _mm256_or_ps(gt123, lt123);
                /*
                if (!inside) {
                    continue;
                }
                */
               const auto isNotInside = _mm256_testz_ps(inside, inside);
               if (isNotInside) {
                   continue;
               }

                //b1 *= denominv;
                //b2 *= denominv;
                //b3 *= denominv;
                b1 = _mm256_mul_ps(b1, denominv);
                b2 = _mm256_mul_ps(b2, denominv);
                b3 = _mm256_mul_ps(b3, denominv);
                //const auto p_ndc_z = b1 * p1_ndc_z + b2 * p2_ndc_z + b3 * p3_ndc_z;
                //const bool depthGE = (fb.zbuf[y*fb.w + x] >= p_ndc_z);
                const __m256 p_ndc_z = _mm256_fmadd_ps(b1, p1_ndc_z, _mm256_fmadd_ps(b2, p2_ndc_z, _mm256_mul_ps(b3, p3_ndc_z)));
                const __m256 depth = _mm256_load_ps(&fb.zbuf[y*fb.w + x]);
                const __m256 depthGE = _mm256_cmp_ps(depth, p_ndc_z, _CMP_GE_OQ);
                /*
                if (!inside || !depthGE) {
                    continue;
                } else {
                    fb.zbuf[y*fb.w + x] = p_ndc_z;
                }
                */
                const auto isLT = _mm256_testz_ps(depthGE, depthGE);
                if (isLT) {
                    continue;
                }
                const __m256 alpha = _mm256_and_ps(inside, depthGE);
                _mm256_store_ps(&fb.zbuf[y*fb.w + x],
                    _mm256_or_ps(
                        _mm256_and_ps(alpha, p_ndc_z),
                        _mm256_andnot_ps(alpha, depth)
                    )
                );

                //const auto n_x = b1 * v1_n_x + b2 * v2_n_x + b3 * v3_n_x;
                //const auto n_y = b1 * v1_n_y + b2 * v2_n_y + b3 * v3_n_y;
                //const auto n_z = b1 * v1_n_z + b2 * v2_n_z + b3 * v3_n_z;
                const __m256 n_x = _mm256_fmadd_ps(b1, v1_n_x, _mm256_fmadd_ps(b2, v2_n_x, _mm256_mul_ps(b3, v3_n_x)));
                const __m256 n_y = _mm256_fmadd_ps(b1, v1_n_y, _mm256_fmadd_ps(b2, v2_n_y, _mm256_mul_ps(b3, v3_n_y)));
                const __m256 n_z = _mm256_fmadd_ps(b1, v1_n_z, _mm256_fmadd_ps(b2, v2_n_z, _mm256_mul_ps(b3, v3_n_z)));

                //const auto c = glm::abs(glm::normalize(n));
                __m256 n = _mm256_fmadd_ps(n_x, n_x, _mm256_fmadd_ps(n_y, n_y, _mm256_mul_ps(n_z, n_z)));
                n = _mm256_rsqrt_ps(n);
                const __m256 c_r = _mm256_andnot_ps(signmask, _mm256_mul_ps(n_x, n));
                const __m256 c_g = _mm256_andnot_ps(signmask, _mm256_mul_ps(n_y, n));
                const __m256 c_b = _mm256_andnot_ps(signmask, _mm256_mul_ps(n_z, n));
                //if (inside && depthGE) {
                //    fb.setPixel(x, y, c);
                //}
                // RGB 8bit * 8pix = 24bytes
                // TODO: rounding (+0.5f)
                __m256i r = _mm256_cvttps_epi32(_mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(c_r, _255), _255), zero));
                __m256i g = _mm256_cvttps_epi32(_mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(c_g, _255), _255), zero));
                __m256i b = _mm256_cvttps_epi32(_mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(c_b, _255), _255), zero));

                // AVX2
                static const __m256i sh32to8R = _mm256_set_epi8(
                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    12,   // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    8,    // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    4,    // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    0,    // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    12,   // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    8,    // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    4,    // R

                    0x80, // A: clear
                    0x80, // B: clear
                    0x80, // G: clear
                    0     // R
                );
                static const __m256i sh32to8G = _mm256_set_epi8(
                    0x80, // A: clear
                    0x80, // B: clear
                    12,   // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    8,    // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    4,    // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    0,    // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    12,   // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    8,    // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    4,    // G
                    0x80, // R: clear

                    0x80, // A: clear
                    0x80, // B: clear
                    0,    // G
                    0x80  // R: clear
                );
                static const __m256i sh32to8B = _mm256_set_epi8(
                    0x80, // A: clear
                    12,   // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    8,    // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    4,    // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    0,    // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    12,   // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    8,    // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    4,    // B
                    0x80, // G: clear
                    0x80, // R: clear

                    0x80, // A: clear
                    0,    // B
                    0x80, // G: clear
                    0x80  // R: clear
                );
                static const __m256i sh32to8A = _mm256_set_epi8(
                    12,   // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    8,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    4,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    0,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    12,   // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    8,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    4,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80, // R: clear

                    0,    // A
                    0x80, // B: clear
                    0x80, // G: clear
                    0x80  // R: clear
                );
                const __m256i rgba8 = _mm256_or_si256(
                    _mm256_shuffle_epi8(r, sh32to8R),
                    _mm256_or_si256(
                        _mm256_shuffle_epi8(g, sh32to8G),
                        _mm256_or_si256(
                            _mm256_shuffle_epi8(b, sh32to8B),
                            _mm256_shuffle_epi8((__m256i)alpha, sh32to8A)
                        )
                    )
                );
                __m256i *pdst = (__m256i *)&fb.buf[fb.w*4*y + 4*(x + 0) + 0];
                const __m256i orig8 = _mm256_load_si256(pdst);
                _mm256_store_si256(pdst,
                    _mm256_or_si256(
                        _mm256_and_si256((__m256i)alpha, rgba8),
                        _mm256_andnot_si256((__m256i)alpha, orig8)
                    )
                );
            }
        }
        #endif
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
    if (!app.setup("raster04_rasterization")) {
        return EXIT_FAILURE;
    }
    app.run([&](int w, int h, const glm::mat4& viewM) -> App::Buf {
        // Raster mode
        enum class RasterMode {
            Shaded,
            Normal,
            Wireframe,
        };
#if 1
        const static auto mode = RasterMode::Normal;
        const static bool animate = true;
        const static bool cullbackface = true;
        const static float fov = 30.f;
        const static float znear = 0.1f;
        const static float zfar = 10.f;
        const static auto view = glm::lookAt(
            glm::vec3(0, 0.8, 1.5), // eye
            glm::vec3(0, 0, 0),     // center
            glm::vec3(0, 1, 0));    // up

        auto *pview = const_cast<glm::mat4 *>(&viewM);
        *pview = view;
#else
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
        ImGui::Checkbox("Enable backface culling", &cullbackface);
        ImGui::Separator();

        // Camera parameters
        static float fov = 30.f;
        static float znear = 0.1f;
        static float zfar = 10.f;
        ImGui::DragFloat("fov", &fov, 0.1f, 0.01f, 180.f);
        ImGui::DragFloat("near", &znear, 0.01f, 0.01f, 10.f);
        ImGui::DragFloat("far", &zfar, 0.01f, 1.f, 1000.f);
        ImGui::Separator();
#endif

        // Framebuffer
        static Framebuffer fb;
        fb.clear(w, h);

        // Transformation matrix
        const auto modelM = animate ? glm::rotate(float(ImGui::GetTime()), glm::vec3(0.f,1.f,0.f)) : glm::mat4(1.f);
        const auto projM = glm::perspective(glm::radians(fov), float(fb.w) / fb.h, znear, zfar);
        const auto transMVP = projM * viewM * modelM;
        const auto transN = glm::mat3(glm::transpose(glm::inverse(modelM)));

        // Direction of light
        const auto lightdir = glm::normalize(glm::vec3(0.5, 0.8, 1));

        // Rasterize
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
                    return glm::vec3(0.2f + 0.8f*glm::max(0.f, glm::dot(n, lightdir)));
                }
                UNREACHABLE();
                return {};
            });

        return { fb.w, fb.h, fb.buf };
    });
    app.shutdown();

    return EXIT_SUCCESS;
}
