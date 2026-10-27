// Apple Neural Engine programs through the private AppleNeuralEngine.framework,
// driven from C++ with the Objective-C runtime's C interface and toll-free
// bridged CoreFoundation objects. Every method is checked against its expected
// Objective-C type encoding before use; a mismatch (a changed framework)
// reports the ANE unavailable instead of calling it with wrong arguments.
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

namespace {

constexpr unsigned kQos = 0x15;  // QOS_CLASS_DEFAULT

struct Pool {
  void* token = objc_autoreleasePoolPush();
  ~Pool() { objc_autoreleasePoolPop(token); }
};

void put_error(char* err, size_t cap, const std::string& message) {
  if (err && cap) snprintf(err, cap, "%s", message.c_str());
}

std::string describe(id object) {
  if (!object) return "(no error)";
  CFStringRef text = CFCopyDescription((CFTypeRef)object);
  char buffer[1024] = {0};
  CFStringGetCString(text, buffer, sizeof buffer, kCFStringEncodingUTF8);
  CFRelease(text);
  return buffer;
}

CFStringRef cfstr(const char* s) { return CFStringCreateWithCString(nullptr, s, kCFStringEncodingUTF8); }

struct Runtime {
  bool ok = false;
  std::string error;
  Class model = nullptr, client_class = nullptr, io_object = nullptr, request = nullptr;
  id client = nullptr;
};

// Type encodings with the frame-offset digits removed.
std::string strip_digits(const char* encoding) {
  std::string out;
  for (const char* p = encoding; p && *p; p++)
    if (*p < '0' || *p > '9') out.push_back(*p);
  return out;
}

bool check(Class cls, bool class_method, const char* selector, const char* expected, std::string& error) {
  SEL sel = sel_registerName(selector);
  Method m = class_method ? class_getClassMethod(cls, sel) : class_getInstanceMethod(cls, sel);
  if (!m) {
    error = std::string("AppleNeuralEngine lacks ") + selector;
    return false;
  }
  const std::string got = strip_digits(method_getTypeEncoding(m));
  if (got != expected) {
    error = std::string("AppleNeuralEngine changed ") + selector + ": " + got + " != " + expected;
    return false;
  }
  return true;
}

Runtime& runtime() {
  static Runtime rt;
  static std::once_flag once;
  std::call_once(once, [] {
    Pool pool;
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      rt.error = std::string("dlopen AppleNeuralEngine: ") + dlerror();
      return;
    }
    rt.model = objc_getClass("_ANEModel");
    rt.client_class = objc_getClass("_ANEClient");
    rt.io_object = objc_getClass("_ANEIOSurfaceObject");
    rt.request = objc_getClass("_ANERequest");
    if (!rt.model || !rt.client_class || !rt.io_object || !rt.request) {
      rt.error = "AppleNeuralEngine classes missing";
      return;
    }
    struct Expect { Class cls; bool cm; const char* sel; const char* enc; };
    const Expect expects[] = {
        {rt.model, true, "modelAtURL:key:", "@@:@@"},
        {rt.model, false, "modelAttributes", "@@:"},
        {rt.client_class, true, "sharedConnection", "@@:"},
        {rt.client_class, false, "compileModel:options:qos:error:", "B@:@@I^@"},
        {rt.client_class, false, "compiledModelExistsFor:", "B@:@"},
        {rt.client_class, false, "purgeCompiledModel:", "v@:@"},
        {rt.client_class, false, "loadModel:options:qos:error:", "B@:@@I^@"},
        {rt.client_class, false, "unloadModel:options:qos:error:", "B@:@@I^@"},
        {rt.client_class, false, "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
        {rt.io_object, true, "objectWithIOSurface:", "@@:^{__IOSurface=}"},
        {rt.request, true,
         "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:",
         "@@:@@@@@@@@@"},
    };
    for (const auto& e : expects)
      if (!check(e.cls, e.cm, e.sel, e.enc, rt.error)) return;
    rt.client = ((id(*)(Class, SEL))objc_msgSend)(rt.client_class, sel_registerName("sharedConnection"));
    if (!rt.client) {
      rt.error = "no ANE client connection";
      return;
    }
    CFRetain((CFTypeRef)rt.client);
    rt.ok = true;
  });
  return rt;
}

struct Program {
  id model = nullptr;
  std::vector<std::string> inputs;  // model input symbol names
  struct Procedure {
    std::string name;
    long id = 0;
    std::vector<long> inputs;   // global input symbol indices, in binding order
    std::vector<long> outputs;  // global output symbol indices
  };
  std::vector<Procedure> procedures;
};

long number(CFTypeRef value) {
  long out = 0;
  if (value && CFGetTypeID(value) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)value, kCFNumberLongType, &out);
  return out;
}

std::vector<long> numbers(CFTypeRef array) {
  std::vector<long> out;
  if (!array || CFGetTypeID(array) != CFArrayGetTypeID()) return out;
  for (CFIndex i = 0; i < CFArrayGetCount((CFArrayRef)array); i++) out.push_back(number(CFArrayGetValueAtIndex((CFArrayRef)array, i)));
  return out;
}

std::string string_of(CFTypeRef value) {
  if (!value || CFGetTypeID(value) != CFStringGetTypeID()) return "";
  char buffer[512] = {0};
  CFStringGetCString((CFStringRef)value, buffer, sizeof buffer, kCFStringEncodingUTF8);
  return buffer;
}

bool compile(Runtime& rt, id model, std::string& error) {
  const void* keys[] = {CFSTR("kANEFModelType"), CFSTR("kANEFNetPlistFilenameKey")};
  const void* values[] = {CFSTR("kANEFModelMIL"), CFSTR("model.mil")};
  CFDictionaryRef options = CFDictionaryCreate(nullptr, keys, values, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  id err = nullptr;
  const bool ok = ((bool(*)(id, SEL, id, id, unsigned, id*))objc_msgSend)(
      rt.client, sel_registerName("compileModel:options:qos:error:"), model, (id)options, kQos, &err);
  CFRelease(options);
  if (!ok) error = "ANE compile failed: " + describe(err);
  return ok;
}

bool load(Runtime& rt, id model, std::string& error) {
  CFDictionaryRef empty = CFDictionaryCreate(nullptr, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  id err = nullptr;
  const bool ok = ((bool(*)(id, SEL, id, id, unsigned, id*))objc_msgSend)(
      rt.client, sel_registerName("loadModel:options:qos:error:"), model, (id)empty, kQos, &err);
  CFRelease(empty);
  if (!ok) error = "ANE load failed: " + describe(err);
  return ok;
}

}  // namespace

extern "C" {

int ironmlx_ane_open(char* err, size_t cap) {
  Runtime& rt = runtime();
  if (!rt.ok) put_error(err, cap, rt.error);
  return rt.ok ? 1 : 0;
}

void* ironmlx_ane_surface_create(size_t rows, size_t width, size_t elem, uint32_t format, size_t* row_bytes,
                                 size_t* alloc_bytes, char* err, size_t cap) {
  const size_t rb = (width * elem + 63) / 64 * 64;
  const size_t alloc = (rb * rows + 16383) / 16384 * 16384;
  auto num = [](long v) { return CFNumberCreate(nullptr, kCFNumberLongType, &v); };
  const void* keys[] = {kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow,
                        kIOSurfaceAllocSize, kIOSurfacePixelFormat};
  CFNumberRef vals[] = {num(long(width)), num(long(rows)), num(long(elem)), num(long(rb)), num(long(alloc)), num(long(format))};
  CFDictionaryRef props = CFDictionaryCreate(nullptr, keys, (const void**)vals, 6, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  for (auto v : vals) CFRelease(v);
  IOSurfaceRef surface = IOSurfaceCreate(props);
  CFRelease(props);
  if (!surface || IOSurfaceGetBytesPerRow(surface) != rb || IOSurfaceGetAllocSize(surface) < alloc) {
    if (surface) CFRelease(surface);
    put_error(err, cap, "IOSurface creation failed");
    return nullptr;
  }
  if (row_bytes) *row_bytes = rb;
  if (alloc_bytes) *alloc_bytes = IOSurfaceGetAllocSize(surface);
  return (void*)surface;
}

void* ironmlx_ane_surface_base(void* surface) { return IOSurfaceGetBaseAddress((IOSurfaceRef)surface); }

void ironmlx_ane_surface_release(void* surface) {
  if (surface) CFRelease((CFTypeRef)surface);
}

// `dir` holds model.mil and weights.bin; `key` names the service's compile cache.
void* ironmlx_ane_program_load(const char* dir, const char* key, char* err, size_t cap) {
  Runtime& rt = runtime();
  if (!rt.ok) {
    put_error(err, cap, rt.error);
    return nullptr;
  }
  Pool pool;
  CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, (const UInt8*)dir, strlen(dir), true);
  CFStringRef k = cfstr(key);
  id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(rt.model, sel_registerName("modelAtURL:key:"), (id)url, (id)k);
  CFRelease(url);
  CFRelease(k);
  if (!model) {
    put_error(err, cap, "ANE model creation failed");
    return nullptr;
  }
  std::string error;
  const bool exists = ((bool(*)(id, SEL, id))objc_msgSend)(rt.client, sel_registerName("compiledModelExistsFor:"), model);
  if (!exists && !compile(rt, model, error)) {
    put_error(err, cap, error);
    return nullptr;
  }
  if (!load(rt, model, error)) {
    // A cached compile the service cannot load: purge, compile again, load again.
    ((void(*)(id, SEL, id))objc_msgSend)(rt.client, sel_registerName("purgeCompiledModel:"), model);
    if (!compile(rt, model, error) || !load(rt, model, error)) {
      put_error(err, cap, error);
      return nullptr;
    }
  }
  auto* program = new Program();
  program->model = model;
  CFRetain((CFTypeRef)model);
  id attrs = ((id(*)(id, SEL))objc_msgSend)(model, sel_registerName("modelAttributes"));
  CFTypeRef desc = attrs ? CFDictionaryGetValue((CFDictionaryRef)attrs, CFSTR("ANEFModelDescription")) : nullptr;
  if (!desc) {
    put_error(err, cap, "ANE model has no description");
    delete program;
    return nullptr;
  }
  CFTypeRef names = CFDictionaryGetValue((CFDictionaryRef)desc, CFSTR("kANEFModelInputSymbolsArrayKey"));
  if (names && CFGetTypeID(names) == CFArrayGetTypeID())
    for (CFIndex i = 0; i < CFArrayGetCount((CFArrayRef)names); i++)
      program->inputs.push_back(string_of(CFArrayGetValueAtIndex((CFArrayRef)names, i)));
  CFTypeRef name_map = CFDictionaryGetValue((CFDictionaryRef)desc, CFSTR("kANEFModelProcedureNameToIDMapKey"));
  CFTypeRef procs = CFDictionaryGetValue((CFDictionaryRef)desc, CFSTR("ANEFModelProcedures"));
  if (procs && CFGetTypeID(procs) == CFArrayGetTypeID()) {
    for (CFIndex i = 0; i < CFArrayGetCount((CFArrayRef)procs); i++) {
      CFDictionaryRef p = (CFDictionaryRef)CFArrayGetValueAtIndex((CFArrayRef)procs, i);
      Program::Procedure proc;
      proc.id = number(CFDictionaryGetValue(p, CFSTR("ANEFModelProcedureID")));
      proc.inputs = numbers(CFDictionaryGetValue(p, CFSTR("ANEFModelInputSymbolIndexArray")));
      proc.outputs = numbers(CFDictionaryGetValue(p, CFSTR("ANEFModelOutputSymbolIndexArray")));
      program->procedures.push_back(proc);
    }
  }
  if (name_map && CFGetTypeID(name_map) == CFDictionaryGetTypeID()) {
    const CFIndex n = CFDictionaryGetCount((CFDictionaryRef)name_map);
    std::vector<const void*> ks(n), vs(n);
    CFDictionaryGetKeysAndValues((CFDictionaryRef)name_map, ks.data(), vs.data());
    for (CFIndex i = 0; i < n; i++)
      for (auto& proc : program->procedures)
        if (proc.id == number(vs[i])) proc.name = string_of(ks[i]);
  }
  return program;
}

int ironmlx_ane_procedure_count(void* program) { return int(((Program*)program)->procedures.size()); }

// Position of the procedure called `name`, or -1.
int ironmlx_ane_procedure_index(void* program, const char* name) {
  auto* p = (Program*)program;
  for (size_t i = 0; i < p->procedures.size(); i++)
    if (p->procedures[i].name == name) return int(i);
  return -1;
}

// Synchronous evaluation of procedure `procedure` with named input surfaces.
int ironmlx_ane_evaluate(void* program, int procedure, const char* const* names, void* const* surfaces, int count,
                         void* output, char* err, size_t cap) {
  Runtime& rt = runtime();
  auto* p = (Program*)program;
  if (procedure < 0 || size_t(procedure) >= p->procedures.size()) {
    put_error(err, cap, "no such ANE procedure");
    return 0;
  }
  const auto& proc = p->procedures[procedure];
  if (proc.outputs.size() != 1) {
    put_error(err, cap, "ANE procedure must have one output");
    return 0;
  }
  Pool pool;
  std::vector<const void*> objects, indices;
  for (long symbol : proc.inputs) {
    const std::string& name = symbol >= 0 && size_t(symbol) < p->inputs.size() ? p->inputs[symbol] : std::string();
    void* surface = nullptr;
    for (int i = 0; i < count; i++)
      if (name == names[i]) surface = surfaces[i];
    if (!surface) {
      put_error(err, cap, "ANE input " + name + " not bound");
      return 0;
    }
    objects.push_back(((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(rt.io_object, sel_registerName("objectWithIOSurface:"), (IOSurfaceRef)surface));
    indices.push_back(CFNumberCreate(nullptr, kCFNumberLongType, &symbol));
  }
  long out_symbol = proc.outputs[0];
  const void* out_object = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(rt.io_object, sel_registerName("objectWithIOSurface:"), (IOSurfaceRef)output);
  CFNumberRef out_index = CFNumberCreate(nullptr, kCFNumberLongType, &out_symbol);
  long proc_id = proc.id;
  CFNumberRef proc_number = CFNumberCreate(nullptr, kCFNumberLongType, &proc_id);
  CFArrayRef in_objects = CFArrayCreate(nullptr, objects.data(), objects.size(), &kCFTypeArrayCallBacks);
  CFArrayRef in_indices = CFArrayCreate(nullptr, indices.data(), indices.size(), &kCFTypeArrayCallBacks);
  CFArrayRef out_objects = CFArrayCreate(nullptr, &out_object, 1, &kCFTypeArrayCallBacks);
  const void* oi = out_index;
  CFArrayRef out_indices = CFArrayCreate(nullptr, &oi, 1, &kCFTypeArrayCallBacks);
  id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
      rt.request,
      sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
      (id)in_objects, (id)in_indices, (id)out_objects, (id)out_indices, nullptr, nullptr, (id)proc_number, nullptr, nullptr);
  CFDictionaryRef empty = CFDictionaryCreate(nullptr, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  id e = nullptr;
  const bool ok = request && ((bool(*)(id, SEL, id, id, id, unsigned, id*))objc_msgSend)(
      rt.client, sel_registerName("evaluateWithModel:options:request:qos:error:"), p->model, (id)empty, request, kQos, &e);
  CFRelease(empty);
  for (auto i : indices) CFRelease(i);
  CFRelease(out_index);
  CFRelease(proc_number);
  CFRelease(in_objects);
  CFRelease(in_indices);
  CFRelease(out_objects);
  CFRelease(out_indices);
  if (!ok) put_error(err, cap, "ANE evaluate failed: " + describe(e));
  return ok ? 1 : 0;
}

void ironmlx_ane_program_release(void* program) {
  auto* p = (Program*)program;
  if (!p) return;
  Runtime& rt = runtime();
  if (rt.ok && p->model) {
    Pool pool;
    CFDictionaryRef empty = CFDictionaryCreate(nullptr, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    id e = nullptr;
    ((bool(*)(id, SEL, id, id, unsigned, id*))objc_msgSend)(rt.client, sel_registerName("unloadModel:options:qos:error:"), p->model, (id)empty, kQos, &e);
    CFRelease(empty);
    CFRelease((CFTypeRef)p->model);
  }
  delete p;
}

}  // extern "C"
