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
    App app;
    if (!app.setup("raster01_window")) {
        return EXIT_FAILURE;
    }
    app.run([&](int w, int h, const glm::mat4& viewM) -> App::Buf {
        static Framebuffer fb;
        fb.clear(w, h);
        const int t = int(ImGui::GetTime()*100.f);
        for (int y = 0; y < fb.h; y++) {
            for (int x = 0; x < fb.w; x++) {
                const int xx = (x+t) / 100;
                const int yy = (y+t) / 100;
                fb.setPixel(x, y, (xx+yy)%2==0 ? glm::vec3(.2f) : glm::vec3(.8f));
            }
        }
        return { fb.w, fb.h, fb.buf.data() };
    });
    app.shutdown();

    return EXIT_SUCCESS;
}
