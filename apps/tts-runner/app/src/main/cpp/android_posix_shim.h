/* Android bionic posix shim — same as bonsai module */
#pragma once
#ifdef __ANDROID__
#ifndef POSIX_MADV_NORMAL
#  define POSIX_MADV_NORMAL     0
#  define POSIX_MADV_RANDOM     1
#  define POSIX_MADV_SEQUENTIAL 2
#  define POSIX_MADV_WILLNEED   3
#  define POSIX_MADV_DONTNEED   4
#  include <stddef.h>
   static inline int posix_madvise(void* /*a*/, size_t /*b*/, int /*c*/) { return 0; }
#endif
#endif
