import re, sys

import sys
# usage: gen_shim.py [path/to/CL/cl_function_types.h]
src = open(sys.argv[1] if len(sys.argv) > 1 else '../opencl-headers/CL/cl_function_types.h').read()

# Match: typedef RETTYPE CL_API_CALL NAME_t( PARAMS );
pat = re.compile(r'typedef\s+([\w \*]+?)\s+CL_API_CALL\s+(\w+)_t\(\s*(.*?)\s*\)\s*;', re.S)

def split_params(s):
    s = s.strip()
    if s == '' or s == 'void':
        return []
    parts = []
    depth = 0
    cur = ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts

def parse_param(p, idx):
    """Returns (decl_text, name) for a parameter."""
    m = re.search(r'\(\s*CL_CALLBACK\s*\*\s*(\w+)\s*\)', p)
    if m:
        return p, m.group(1)
    # type-only param documented via a /** ... */ comment (e.g. QCOM recording APIs)
    m = re.match(r'^(.*?)\s*/\*\*.*?\*/\s*$', p)
    if m:
        synth = f'arg{idx}'
        return f'{m.group(1).strip()} {synth}', synth
    # strip array brackets
    p2 = re.sub(r'\[\s*\]\s*$', '', p).strip()
    m = re.search(r'(\w+)$', p2)
    if m:
        return p, m.group(1)
    return None, None

# These reference QCOM perf-monitor / recording / EGL-interop types that
# aren't present in our header set (newer extensions our SDK snapshot lacks)
# and aren't used by ggml's OpenCL backend.
SKIP = {
    'clCreateBufferFromImageQCOM', 'clGetPerfMonitorGroupInfoQCOM',
    'clGetPerfMonitorCounterInfoQCOM', 'clCreatePerfMonitorQCOM',
    'clRetainPerfMonitorQCOM', 'clReleasePerfMonitorQCOM',
    'clEnqueueBeginPerfMonitorQCOM', 'clEnqueueEndPerfMonitorQCOM',
    'clEnqueueReadPerfMonitorQCOM', 'clGetPerfMonitorInfoQCOM',
    'clGetDeviceImageInfoQCOM', 'clQueryImageInfoQCOM',
    'clCreateFromEGLImageIMG', 'clNewRecordingQCOM', 'clEndRecordingQCOM',
    'clReleaseRecordingQCOM', 'clRetainRecordingQCOM',
    'clEnqueueRecordingQCOM', 'clEnqueueRecordingSVMQCOM',
    'clSetPerfHintQCOM',
}

functions = []
for m in pat.finditer(src):
    rettype = m.group(1).strip()
    name = m.group(2)
    if name in SKIP:
        continue
    params_raw = m.group(3)
    raw_params = split_params(params_raw)
    decls, names = [], []
    ok = True
    for i, p in enumerate(raw_params):
        d, n = parse_param(p, i)
        if n is None:
            ok = False
            break
        decls.append(d)
        names.append(n)
    if not ok:
        print("WARN: could not parse param name in", name, raw_params, file=sys.stderr)
        continue
    functions.append((rettype, name, decls, names))

print(f"// Auto-generated OpenCL dlopen-redirect shim — {len(functions)} functions", )
print('// Bridges to the real vendor Adreno driver (libOpenCL.so in /vendor/lib64) at runtime')
print('// via dlopen("libOpenCL.so", ...) + dlsym, so the app can statically link against')
print('// the OpenCL ABI without bundling proprietary vendor code.')
print('#include <dlfcn.h>')
print('#include <android/log.h>')
print('#define CL_TARGET_OPENCL_VERSION 300')
print('#include <CL/cl.h>')
print('#include <CL/cl_ext.h>')
print('// NOTE: cl_function_types.h is NOT included — it declares typedefs for newer')
print('// QCOM perf-monitor/recording/EGL extensions whose types our header snapshot')
print('// lacks. We declare our own _t typedefs below for just the functions we shim.')
print()
print('#define TAG "OpenCLShim"')
print()
print('static void* real_handle() {')
print('    static void* h = []() -> void* {')
print('        void* h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);')
print('        if (!h) h = dlopen("libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);')
print('        __android_log_print(h ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, TAG,')
print('                            "real driver dlopen %s: %s", h ? "OK" : "FAILED", dlerror());')
print('        return h;')
print('    }();')
print('    return h;')
print('}')
print()
print('static void* resolve(const char* name) {')
print('    void* h = real_handle();')
print('    if (!h) return nullptr;')
print('    void* sym = dlsym(h, name);')
print('    if (!sym) __android_log_print(ANDROID_LOG_WARN, TAG, "missing symbol: %s", name);')
print('    return sym;')
print('}')
print()
print('extern "C" {')
print()

for rettype, name, decls, names in functions:
    decl_params = ', '.join(decls) if decls else 'void'
    call_args = ', '.join(names)
    is_void = rettype.strip() == 'void'
    print(f'typedef {rettype} (CL_API_CALL *{name}_fn_t)({decl_params});')
    print(f'CL_API_ENTRY {rettype} CL_API_CALL {name}({decl_params}) {{')
    print(f'    using fn_t = {name}_fn_t;')
    print(f'    static fn_t fn = (fn_t)resolve("{name}");')
    if is_void:
        print('    if (!fn) return;')
        print(f'    fn({call_args});')
    else:
        print(f'    if (!fn) return ({rettype})0;')
        print(f'    return fn({call_args});')
    print('}')
    print()

print('} // extern "C"')
