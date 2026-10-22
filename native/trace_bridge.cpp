// Rust model spans share the NPU library's Perfetto categories and session.
#include "rk_npu_trace.h"
#include <cstdio>
#include <exception>

extern "C" int rknpu2_trace_start(const char * path) {
    try { rk_npu_trace::start(path); return 0; }
    catch (const std::exception & e) { std::fprintf(stderr,"Perfetto: %s\n",e.what()); return -1; }
}
extern "C" int rknpu2_trace_stop() {
    try { rk_npu_trace::stop(); return 0; }
    catch (const std::exception & e) { std::fprintf(stderr,"Perfetto: %s\n",e.what()); return -1; }
}
extern "C" void rknpu2_trace_begin(const char * name, int layer, int rows, int position) {
    RK_NPU_TRACE_BEGIN("llm",perfetto::DynamicString(name),"layer",layer,"rows",rows,"position",position);
}
extern "C" void rknpu2_trace_end() { RK_NPU_TRACE_END("llm"); }
