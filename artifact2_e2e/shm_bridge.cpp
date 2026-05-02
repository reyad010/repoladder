#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <cstring>
#include <sys/stat.h>
#include <map>
#include <string>
//#include <sched.h>

#ifdef __x86_64__
  #include <immintrin.h>
  static inline void cpu_relax(){ _mm_pause(); }
#else
  static inline void cpu_relax(){ sched_yield(); }
#endif

extern "C" {

struct ShmHeader {
    int32_t  status;      // 0=idle, 1=ready
    uint32_t size_f;      // number of float elements valid in data[]
    uint32_t capacity_f;  // capacity of data[] in float elements
    uint32_t reserved;    // padding
    uint8_t  pad[48];     // pad header to 64B
    unsigned char data[1];// payload (float[])
};

struct ShmCtx {
    int fd{-1};
    size_t bytes{0};
    ShmHeader* hdr{nullptr};
    bool is_owner{false};
};

static std::map<std::string, ShmCtx> g_ctx_map;

int shm_close_named(const char* name, int do_unlink);
int shm_unlink_named(const char* name);

static inline void store_status(volatile int32_t* p, int v){
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline int load_status(volatile int32_t* p){
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

static const char* normalize_name(const char* in, char* out, size_t outsz){
    // Accept "/name" or "/dev/shm/name"; normalize to "/name"
    if (!in || !*in) return nullptr;
    const char* name = in;
    const char prefix[] = "/dev/shm/";
    if (std::strncmp(in, prefix, sizeof(prefix)-1) == 0) {
        name = in + (sizeof(prefix)-1);
    }
    // Ensure single leading slash
    if (*name == '/') {
        std::snprintf(out, outsz, "%s", name);
    } else {
        std::snprintf(out, outsz, "/%s", name);
    }
    return out;
}

static bool shm_map_named(const char* raw_name, size_t capacity_floats, bool owner){
    char namebuf[256];
    const char* shm_name = normalize_name(raw_name, namebuf, sizeof(namebuf));
    if (!shm_name) { std::fprintf(stderr, "invalid shm name\n"); return false; }

    std::string key(shm_name);
    
    if (g_ctx_map.find(key) != g_ctx_map.end()) {
        return true;
    }

    int fd = shm_open(shm_name, owner ? (O_CREAT|O_RDWR) : O_RDWR, 0666);
//    if (fd < 0){ std::perror("shm_open"); return false; }
    if (fd < 0){ return false; }

    size_t bytes = 0;

    if (owner){
        bytes = sizeof(ShmHeader) - 1 + capacity_floats*sizeof(float);
        if (ftruncate(fd, (off_t)bytes) != 0){
            std::perror("ftruncate");
            close(fd);
            shm_unlink(shm_name);
            return false;
        }
    } else {
        struct stat st{};
        if (fstat(fd, &st) != 0){
            std::perror("fstat");
            close(fd);
            return false;
        }
        if (st.st_size <= 0){
            std::fprintf(stderr, "shm size <= 0\n");
            close(fd);
            return false;
        }
        bytes = (size_t)st.st_size;
    }

    void* addr = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED){
        int e = errno; close(fd); errno = e; std::perror("mmap"); return false;
    }

    ShmCtx ctx;
    ctx.fd = fd;
    ctx.bytes = bytes;
    ctx.hdr = reinterpret_cast<ShmHeader*>(addr);
    ctx.is_owner = owner;

    if (owner){
        ctx.hdr->status = 0;
        ctx.hdr->size_f = 0;
        ctx.hdr->capacity_f = (uint32_t)capacity_floats;
    }
    
    g_ctx_map[key] = ctx;
    return true;
}

static ShmCtx* get_ctx(const char* raw_name){
    char namebuf[256];
    const char* shm_name = normalize_name(raw_name, namebuf, sizeof(namebuf));
    if (!shm_name) return nullptr;
    
    std::string key(shm_name);
    auto it = g_ctx_map.find(key);
    if (it == g_ctx_map.end()) return nullptr;
    return &(it->second);
}

int shm_owner_create_named(const char* name, uint32_t capacity_floats){
    return shm_map_named(name, capacity_floats, true) ? 0 : -1;
}
int shm_client_open_named(const char* name, uint32_t capacity_floats){
    (void)capacity_floats; // capacity is enforced by owner
    return shm_map_named(name, 0, false) ? 0 : -1;
}
void shm_close_all(){
    for (auto& pair : g_ctx_map) {
        ShmCtx& ctx = pair.second;
        if (ctx.hdr) {
            munmap(ctx.hdr, ctx.bytes);
            ctx.hdr = nullptr;
        }
        if (ctx.fd >= 0) {
            close(ctx.fd);
            ctx.fd = -1;
        }
    }
    g_ctx_map.clear();
}

static void wait_idle(ShmCtx* ctx){
    while (load_status(&ctx->hdr->status) != 0){
        cpu_relax();
    }
}

// Get a writable pointer to the payload buffer (size in floats returned via capacity)
float* shm_get_write_ptr_named(const char* name, uint32_t* capacity_floats){
    ShmCtx* ctx = get_ctx(name);
    if (!ctx || !ctx->hdr) return nullptr;
    wait_idle(ctx);
    if (capacity_floats) *capacity_floats = ctx->hdr->capacity_f;
    return reinterpret_cast<float*>(ctx->hdr->data);
}

// After writing, call this to publish n floats as ready.
// n==0 is treated as an end-of-stream sentinel.
int shm_publish_done_named(const char* name, uint32_t n_floats){
    ShmCtx* ctx = get_ctx(name);
    if (!ctx || !ctx->hdr) return -1;
    if (n_floats > ctx->hdr->capacity_f) return -2;
    ctx->hdr->size_f = n_floats;
    store_status(&ctx->hdr->status, 1);
    return 0;
}

// ---- consumer side (read-only pointer) ----
float* shm_wait_ready_get_ptr_named(const char* name, int timeout_ms){
    ShmCtx* ctx = get_ctx(name);
    if (!ctx || !ctx->hdr) return nullptr;
    const int sleep_us = 1;
    int waited_ms = 0;
    while (load_status(&ctx->hdr->status) != 1){
        if (timeout_ms >= 0 && waited_ms >= timeout_ms) return nullptr;
        usleep(sleep_us);
        waited_ms += sleep_us / 1000;
        //sched_yield();
    }
    return reinterpret_cast<float*>(ctx->hdr->data);
}

uint32_t shm_size_named(const char* name){
    ShmCtx* ctx = get_ctx(name);
    return (ctx && ctx->hdr) ? ctx->hdr->size_f : 0;
}

uint32_t shm_capacity_named(const char* name){
    ShmCtx* ctx = get_ctx(name);
    return (ctx && ctx->hdr) ? ctx->hdr->capacity_f : 0;
}

void shm_ack_named(const char* name){
    ShmCtx* ctx = get_ctx(name);
    if (ctx && ctx->hdr) {
        store_status(&ctx->hdr->status, 0);
    }
}

int shm_close_named(const char* raw_name, int do_unlink) {
    char namebuf[256];
    const char* shm_name = normalize_name(raw_name, namebuf, sizeof(namebuf));
    if (!shm_name) return -1;

    std::string key(shm_name);
    auto it = g_ctx_map.find(key);
    if (it != g_ctx_map.end()) {
        ShmCtx& ctx = it->second;
        if (ctx.hdr) {
            munmap(ctx.hdr, ctx.bytes);
            ctx.hdr = nullptr;
        }
        if (ctx.fd >= 0) {
            close(ctx.fd);
            ctx.fd = -1;
        }
        g_ctx_map.erase(it);
    }
    if (do_unlink) {
        // Either side may unlink; object is removed when last fd closes
        shm_unlink(shm_name);
    }
    return 0;
}

int shm_unlink_named(const char* raw_name) {
    char namebuf[256];
    const char* shm_name = normalize_name(raw_name, namebuf, sizeof(namebuf));
    if (!shm_name) return -1;
    return shm_unlink(shm_name);
}

static const char* DEFAULT_SHM_NAME = "/shm_bridge_default";

int shm_owner_create(uint32_t capacity_floats){
    return shm_owner_create_named(DEFAULT_SHM_NAME, capacity_floats);
}

int shm_client_open(uint32_t capacity_floats){
    return shm_client_open_named(DEFAULT_SHM_NAME, capacity_floats);
}

float* shm_get_write_ptr(uint32_t* capacity_floats){
    return shm_get_write_ptr_named(DEFAULT_SHM_NAME, capacity_floats);
}

int shm_publish_done(uint32_t n_floats){
    return shm_publish_done_named(DEFAULT_SHM_NAME, n_floats);
}

float* shm_wait_ready_get_ptr(int timeout_ms){
    return shm_wait_ready_get_ptr_named(DEFAULT_SHM_NAME, timeout_ms);
}

uint32_t shm_size(){
    return shm_size_named(DEFAULT_SHM_NAME);
}

uint32_t shm_capacity(){
    return shm_capacity_named(DEFAULT_SHM_NAME);
}

void shm_ack(){
    shm_ack_named(DEFAULT_SHM_NAME);
}

} // extern "C"