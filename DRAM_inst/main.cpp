#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#define ERT_FLOP 2 
#define ERT_TRIALS_MIN 1
#define ERT_WORKING_SET_MIN 1
#define GBUNIT (1024 * 1024 * 1024)

/* SPDX-License-Identifier: GPL-2.0 */
/*
 * BRIEF DESCRIPTION
 *
 * Header for commonly used defines
 *
 * Copyright 2019 Regents of the University of California,
 * UCSD Non-Volatile Systems Lab
 */

// #ifdef __KERNEL__
#if 0
  #include <asm/fpu/api.h>
  #define KERNEL_BEGIN \
	 kernel_fpu_begin();
  #define KERNEL_END \
	 kernel_fpu_end();
#else
  #define KERNEL_BEGIN do { } while (0);
  #define KERNEL_END do { } while (0);
#endif

/* Assembly instructions utilize the following registers:
 * rsi: memory address
 * rax, rdx, rcx, r8d and r9d: timing
 * rdx: populating cache-lines
 * ymm0: streaming instructions
 */
#define REGISTERS "rsi", "rax", "rdx", "rcx", "r8", "r9", "ymm0"
#define REGISTERS_NONSSE "rsi", "rax", "rdx", "rcx", "r8", "r9"

/* ymm0: 256-bit register (requires AVX support)
 * vbroadcastsd: VEX.256-bit version (r[0] = r[1] = r[2] = r[3] = v)
 */
#define LOAD_VALUE              "vbroadcastsd %[value], %%ymm0 \n"
#define LOAD_ADDR               "mov %[memarea], %%rsi \n" \
                                "mfence \n"

#define FLUSH_CACHE_LINE        "clflush 0*32(%%rsi) \n" \
                                "clflush 2*32(%%rsi) \n" \
                                "clflush 4*32(%%rsi) \n" \
                                "clflush 6*32(%%rsi) \n" \
                                "mfence \n"

#define LOAD_CACHE_LINE         "movq 0*8(%%rsi), %%rdx \n" \
                                "movq 1*8(%%rsi), %%rdx \n" \
                                "movq 2*8(%%rsi), %%rdx \n" \
                                "movq 3*8(%%rsi), %%rdx \n" \
                                "movq 4*8(%%rsi), %%rdx \n" \
                                "movq 5*8(%%rsi), %%rdx \n" \
                                "movq 6*8(%%rsi), %%rdx \n" \
                                "movq 7*8(%%rsi), %%rdx \n" \
                                "movq 8*8(%%rsi), %%rdx \n" \
                                "movq 9*8(%%rsi), %%rdx \n" \
                                "movq 10*8(%%rsi), %%rdx \n" \
                                "movq 11*8(%%rsi), %%rdx \n" \
                                "movq 12*8(%%rsi), %%rdx \n" \
                                "movq 13*8(%%rsi), %%rdx \n" \
                                "movq 14*8(%%rsi), %%rdx \n" \
                                "movq 15*8(%%rsi), %%rdx \n" \
                                "movq 16*8(%%rsi), %%rdx \n" \
                                "movq 17*8(%%rsi), %%rdx \n" \
                                "movq 18*8(%%rsi), %%rdx \n" \
                                "movq 19*8(%%rsi), %%rdx \n" \
                                "movq 20*8(%%rsi), %%rdx \n" \
                                "movq 21*8(%%rsi), %%rdx \n" \
                                "movq 22*8(%%rsi), %%rdx \n" \
                                "movq 23*8(%%rsi), %%rdx \n" \
                                "movq 24*8(%%rsi), %%rdx \n" \
                                "movq 25*8(%%rsi), %%rdx \n" \
                                "movq 26*8(%%rsi), %%rdx \n" \
                                "movq 27*8(%%rsi), %%rdx \n" \
                                "movq 28*8(%%rsi), %%rdx \n" \
                                "movq 29*8(%%rsi), %%rdx \n" \
                                "movq 30*8(%%rsi), %%rdx \n" \
                                "movq 31*8(%%rsi), %%rdx \n" \
                                "mfence \n"

#define CLEAR_PIPELINE          "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n" \
                                "nop \nnop \nnop \nnop \nnop \nnop \n"

/* rdtscp: reads current timestamp to EDX:EAX and also sets ECX
 * higher 32-bits of RAX, RDX and RCX are cleared
 */
#define TIMING_BEG              "rdtscp \n" \
                                "lfence \n" \
                                "mov %%edx, %%r9d \n" \
                                "mov %%eax, %%r8d \n"

/* r9d = old EDX
 * r8d = old EAX
 * Here is what we do to compute t1 and t2:
 * - RDX holds t2
 * - RAX holds t1
 */
#define TIMING_END              "mfence \n" \
                                "rdtscp \n" \
                                "lfence \n" \
                                "shl $32, %%rdx \n" \
                                "or  %%rax, %%rdx \n" \
                                "mov %%r9d, %%eax \n" \
                                "shl $32, %%rax \n" \
                                "or  %%r8, %%rax \n" \
                                "mov %%rax, %[t1] \n" \
                                "mov %%rdx, %[t2] \n"

#define TIMING_END_NOFENCE      "rdtscp \n" \
                                "shl $32, %%rdx \n" \
                                "or  %%rax, %%rdx \n" \
                                "mov %%r9d, %%eax \n" \
                                "shl $32, %%rax \n" \
                                "or  %%r8, %%rax \n" \
                                "mov %%rax, %[t1] \n" \
                                "mov %%rdx, %[t2] \n"

/*
 * 64-byte benchmarks
 */
uint64_t store_64byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    // vmovdqa: 32-byte store to memory
    asm volatile(LOAD_VALUE
        LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_64byte_clflush(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflush (%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_64byte_clwb(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clwb (%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_64byte_clflushopt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflushopt (%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t nstore_64byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    /*
     * vmovntpd: 32-byte non-temporal store (check below)
     * https://software.intel.com/en-us/node/524246
     */
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntpd %%ymm0, 0*32(%%rsi) \n"
        "vmovntpd %%ymm0, 1*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_64byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovdqa 1*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_64byte_fence_nt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    /*
     * Requires avx2
     * https://www.felixcloutier.com/x86/MOVNTDQA.html
     */
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovntdqa 1*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}


uint64_t baseline(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(CLEAR_PIPELINE
        TIMING_BEG
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        :: REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_64byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_64byte_clflush_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        "clflush (%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_64byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq 0*8(%%rsi), %%rdx \n"
        "movq 1*8(%%rsi), %%rdx \n"
        "movq 2*8(%%rsi), %%rdx \n"
        "movq 3*8(%%rsi), %%rdx \n"
        "movq 4*8(%%rsi), %%rdx \n"
        "movq 5*8(%%rsi), %%rdx \n"
        "movq 6*8(%%rsi), %%rdx \n"
        "movq 7*8(%%rsi), %%rdx \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t(*latency_funcs_64byte[])(char *) = {
    &load_64byte_fence, // load + fence
    &load_64byte_fence_nt, // non-temporal load + fence
    &store_64byte_fence, // store + fence
    &store_64byte_clflush, // store + clflush
    &store_64byte_clwb, // store + clwb
    &store_64byte_clflushopt, // store + clflushopt
    &nstore_64byte_fence, // non-temporal store + fence
    &store_64byte_fence_movq, // store + fence (movq)
    &store_64byte_clflush_movq, // store - clflush (movq)
    &load_64byte_fence_movq, // load + fence (movq)
    &baseline
};

/*
 * 128-byte benchmarks
 */
uint64_t store_128byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_VALUE
        LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_128byte_clflush(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflush (%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clflush 2*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_128byte_clwb(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clwb (%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clwb 2*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_128byte_clflushopt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflushopt (%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clflushopt 2*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t nstore_128byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntpd %%ymm0, 0*32(%%rsi) \n"
        "vmovntpd %%ymm0, 1*32(%%rsi) \n"
        "vmovntpd %%ymm0, 2*32(%%rsi) \n"
        "vmovntpd %%ymm0, 3*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_128byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovdqa 1*32(%%rsi), %%ymm1 \n"
        "vmovdqa 2*32(%%rsi), %%ymm1 \n"
        "vmovdqa 3*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_128byte_fence_nt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovntdqa 1*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 2*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 3*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_128byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        "movq %%rdx, 8*8(%%rsi) \n"
        "movq %%rdx, 9*8(%%rsi) \n"
        "movq %%rdx, 10*8(%%rsi) \n"
        "movq %%rdx, 11*8(%%rsi) \n"
        "movq %%rdx, 12*8(%%rsi) \n"
        "movq %%rdx, 13*8(%%rsi) \n"
        "movq %%rdx, 14*8(%%rsi) \n"
        "movq %%rdx, 15*8(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_128byte_clflush_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        "clflush (%%rsi) \n"
        "movq %%rdx, 8*8(%%rsi) \n"
        "movq %%rdx, 9*8(%%rsi) \n"
        "movq %%rdx, 10*8(%%rsi) \n"
        "movq %%rdx, 11*8(%%rsi) \n"
        "movq %%rdx, 12*8(%%rsi) \n"
        "movq %%rdx, 13*8(%%rsi) \n"
        "movq %%rdx, 14*8(%%rsi) \n"
        "movq %%rdx, 15*8(%%rsi) \n"
        "clflush 8*8(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_128byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq 0*8(%%rsi), %%rdx \n"
        "movq 1*8(%%rsi), %%rdx \n"
        "movq 2*8(%%rsi), %%rdx \n"
        "movq 3*8(%%rsi), %%rdx \n"
        "movq 4*8(%%rsi), %%rdx \n"
        "movq 5*8(%%rsi), %%rdx \n"
        "movq 6*8(%%rsi), %%rdx \n"
        "movq 7*8(%%rsi), %%rdx \n"
        "movq 8*8(%%rsi), %%rdx \n"
        "movq 9*8(%%rsi), %%rdx \n"
        "movq 10*8(%%rsi), %%rdx \n"
        "movq 11*8(%%rsi), %%rdx \n"
        "movq 12*8(%%rsi), %%rdx \n"
        "movq 13*8(%%rsi), %%rdx \n"
        "movq 14*8(%%rsi), %%rdx \n"
        "movq 15*8(%%rsi), %%rdx \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t(*latency_funcs_128byte[])(char *) = {
    &load_128byte_fence,
    &load_128byte_fence_nt,
    &store_128byte_fence,
    &store_128byte_clflush,
    &store_128byte_clwb,
    &store_128byte_clflushopt,
    &nstore_128byte_fence,
    &store_128byte_fence_movq,
    &store_128byte_clflush_movq,
    &load_128byte_fence_movq,
    &baseline
};

/*
 * 256-byte benchmarks
 */
uint64_t store_256byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_VALUE
        LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "vmovdqa %%ymm0, 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 5*32(%%rsi) \n"
        "vmovdqa %%ymm0, 6*32(%%rsi) \n"
        "vmovdqa %%ymm0, 7*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}



uint64_t store_256byte_clflush(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflush 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clflush 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 5*32(%%rsi) \n"
        "clflush 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 6*32(%%rsi) \n"
        "vmovdqa %%ymm0, 7*32(%%rsi) \n"
        "clflush 6*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_256byte_clwb(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clwb 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clwb 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 5*32(%%rsi) \n"
        "clwb 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 6*32(%%rsi) \n"
        "vmovdqa %%ymm0, 7*32(%%rsi) \n"
        "clwb 6*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_256byte_clflushopt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa %%ymm0, 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 1*32(%%rsi) \n"
        "clflushopt 0*32(%%rsi) \n"
        "vmovdqa %%ymm0, 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 3*32(%%rsi) \n"
        "clflushopt 2*32(%%rsi) \n"
        "vmovdqa %%ymm0, 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 5*32(%%rsi) \n"
        "clflushopt 4*32(%%rsi) \n"
        "vmovdqa %%ymm0, 6*32(%%rsi) \n"
        "vmovdqa %%ymm0, 7*32(%%rsi) \n"
        "clflushopt 6*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t nstore_256byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_VALUE
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntpd %%ymm0, 0*32(%%rsi) \n"
        "vmovntpd %%ymm0, 1*32(%%rsi) \n"
        "vmovntpd %%ymm0, 2*32(%%rsi) \n"
        "vmovntpd %%ymm0, 3*32(%%rsi) \n"
        "vmovntpd %%ymm0, 4*32(%%rsi) \n"
        "vmovntpd %%ymm0, 5*32(%%rsi) \n"
        "vmovntpd %%ymm0, 6*32(%%rsi) \n"
        "vmovntpd %%ymm0, 7*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_256byte_fence(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovdqa 1*32(%%rsi), %%ymm1 \n"
        "vmovdqa 2*32(%%rsi), %%ymm1 \n"
        "vmovdqa 3*32(%%rsi), %%ymm1 \n"
        "vmovdqa 4*32(%%rsi), %%ymm1 \n"
        "vmovdqa 5*32(%%rsi), %%ymm1 \n"
        "vmovdqa 6*32(%%rsi), %%ymm1 \n"
        "vmovdqa 7*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_256byte_fence_nt(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "vmovntdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovntdqa 1*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 2*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 3*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 4*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 5*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 6*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 7*32(%%rsi), %%ymm1 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_256byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        "movq %%rdx, 8*8(%%rsi) \n"
        "movq %%rdx, 9*8(%%rsi) \n"
        "movq %%rdx, 10*8(%%rsi) \n"
        "movq %%rdx, 11*8(%%rsi) \n"
        "movq %%rdx, 12*8(%%rsi) \n"
        "movq %%rdx, 13*8(%%rsi) \n"
        "movq %%rdx, 14*8(%%rsi) \n"
        "movq %%rdx, 15*8(%%rsi) \n"
        "movq %%rdx, 16*8(%%rsi) \n"
        "movq %%rdx, 17*8(%%rsi) \n"
        "movq %%rdx, 18*8(%%rsi) \n"
        "movq %%rdx, 19*8(%%rsi) \n"
        "movq %%rdx, 20*8(%%rsi) \n"
        "movq %%rdx, 21*8(%%rsi) \n"
        "movq %%rdx, 22*8(%%rsi) \n"
        "movq %%rdx, 23*8(%%rsi) \n"
        "movq %%rdx, 24*8(%%rsi) \n"
        "movq %%rdx, 25*8(%%rsi) \n"
        "movq %%rdx, 26*8(%%rsi) \n"
        "movq %%rdx, 27*8(%%rsi) \n"
        "movq %%rdx, 28*8(%%rsi) \n"
        "movq %%rdx, 29*8(%%rsi) \n"
        "movq %%rdx, 30*8(%%rsi) \n"
        "movq %%rdx, 31*8(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t store_256byte_clflush_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        LOAD_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq %%rdx, 0*8(%%rsi) \n"
        "movq %%rdx, 1*8(%%rsi) \n"
        "movq %%rdx, 2*8(%%rsi) \n"
        "movq %%rdx, 3*8(%%rsi) \n"
        "movq %%rdx, 4*8(%%rsi) \n"
        "movq %%rdx, 5*8(%%rsi) \n"
        "movq %%rdx, 6*8(%%rsi) \n"
        "movq %%rdx, 7*8(%%rsi) \n"
        "clflush 0*8(%%rsi) \n"
        "movq %%rdx, 8*8(%%rsi) \n"
        "movq %%rdx, 9*8(%%rsi) \n"
        "movq %%rdx, 10*8(%%rsi) \n"
        "movq %%rdx, 11*8(%%rsi) \n"
        "movq %%rdx, 12*8(%%rsi) \n"
        "movq %%rdx, 13*8(%%rsi) \n"
        "movq %%rdx, 14*8(%%rsi) \n"
        "movq %%rdx, 15*8(%%rsi) \n"
        "clflush 8*8(%%rsi) \n"
        "movq %%rdx, 16*8(%%rsi) \n"
        "movq %%rdx, 17*8(%%rsi) \n"
        "movq %%rdx, 18*8(%%rsi) \n"
        "movq %%rdx, 19*8(%%rsi) \n"
        "movq %%rdx, 20*8(%%rsi) \n"
        "movq %%rdx, 21*8(%%rsi) \n"
        "movq %%rdx, 22*8(%%rsi) \n"
        "movq %%rdx, 23*8(%%rsi) \n"
        "clflush 16*8(%%rsi) \n"
        "movq %%rdx, 24*8(%%rsi) \n"
        "movq %%rdx, 25*8(%%rsi) \n"
        "movq %%rdx, 26*8(%%rsi) \n"
        "movq %%rdx, 27*8(%%rsi) \n"
        "movq %%rdx, 28*8(%%rsi) \n"
        "movq %%rdx, 29*8(%%rsi) \n"
        "movq %%rdx, 30*8(%%rsi) \n"
        "movq %%rdx, 31*8(%%rsi) \n"
        "clflush 24*8(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t load_256byte_fence_movq(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    KERNEL_BEGIN
    asm volatile(LOAD_ADDR
        FLUSH_CACHE_LINE
        CLEAR_PIPELINE
        TIMING_BEG
        "movq 0*8(%%rsi), %%rdx \n"
        "movq 1*8(%%rsi), %%rdx \n"
        "movq 2*8(%%rsi), %%rdx \n"
        "movq 3*8(%%rsi), %%rdx \n"
        "movq 4*8(%%rsi), %%rdx \n"
        "movq 5*8(%%rsi), %%rdx \n"
        "movq 6*8(%%rsi), %%rdx \n"
        "movq 7*8(%%rsi), %%rdx \n"
        "movq 8*8(%%rsi), %%rdx \n"
        "movq 9*8(%%rsi), %%rdx \n"
        "movq 10*8(%%rsi), %%rdx \n"
        "movq 11*8(%%rsi), %%rdx \n"
        "movq 12*8(%%rsi), %%rdx \n"
        "movq 13*8(%%rsi), %%rdx \n"
        "movq 14*8(%%rsi), %%rdx \n"
        "movq 15*8(%%rsi), %%rdx \n"
        "movq 16*8(%%rsi), %%rdx \n"
        "movq 17*8(%%rsi), %%rdx \n"
        "movq 18*8(%%rsi), %%rdx \n"
        "movq 19*8(%%rsi), %%rdx \n"
        "movq 20*8(%%rsi), %%rdx \n"
        "movq 21*8(%%rsi), %%rdx \n"
        "movq 22*8(%%rsi), %%rdx \n"
        "movq 23*8(%%rsi), %%rdx \n"
        "movq 24*8(%%rsi), %%rdx \n"
        "movq 25*8(%%rsi), %%rdx \n"
        "movq 26*8(%%rsi), %%rdx \n"
        "movq 27*8(%%rsi), %%rdx \n"
        "movq 28*8(%%rsi), %%rdx \n"
        "movq 29*8(%%rsi), %%rdx \n"
        "movq 30*8(%%rsi), %%rdx \n"
        "movq 31*8(%%rsi), %%rdx \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr)
        : REGISTERS_NONSSE);
    KERNEL_END
    return t2 - t1;
}

uint64_t(*latency_funcs_256byte[])(char *) = {
    &load_256byte_fence,
    &load_256byte_fence_nt,
    &store_256byte_fence,
    &store_256byte_clflush,
    &store_256byte_clwb,
    &store_256byte_clflushopt,
    &nstore_256byte_fence,
    &store_256byte_fence_movq,
    &store_256byte_clflush_movq,
    &load_256byte_fence_movq,
    &baseline
};

// Benchmark functions map
const char *latency_bench_map[] = {
    "load-fence",
    "ntload-fence",
    "store-fence",
    "store-clflush",
    "store-clwb",
    "store-clflushopt",
    "nstore-fence",
    "store-fence-movq",
    "store-clflush-movq",
    "load-fence-movq",
    "baseline"
};

// Kernel-level task lists

enum latency_tasks {
    load_fence_64 = 0,
    ntload_fence_64,
    store_fence_64,
    store_clflush_64,
#ifdef AEP_SUPPORTED
    store_clwb_64,
    store_clflushopt_64,
#endif
    nstore_fence_64,
    store_fence_movq_64,
    store_clflush_movq_64,
    load_fence_movq_64,

    load_fence_128,
    ntload_fence_128,
    store_fence_128,
    store_clflush_128,
#ifdef AEP_SUPPORTED
    store_clwb_128,
    store_clflushopt_128,
#endif
    nstore_fence_128,
    store_fence_movq_128,
    store_clflush_movq_128,
    load_fence_movq_128,

    load_fence_256,
    ntload_fence_256,
    store_fence_256,
    store_clflush_256,
#ifdef AEP_SUPPORTED
    store_clwb_256,
    store_clflushopt_256,
#endif
    nstore_fence_256,
    store_fence_movq_256,
    store_clflush_movq_256,
    load_fence_movq_256,
    task_baseline,

    BASIC_OPS_TASK_COUNT
};

static const int latency_tasks_skip[BASIC_OPS_TASK_COUNT] = {
64,
64,
64,
64,
#ifdef AEP_SUPPORTED
64,
64,
#endif
64,
64,
64,
64,

128,
128,
128,
128,
#ifdef AEP_SUPPORTED
128,
128,
#endif
128,
128,
128,
128,

256,
256,
256,
256,
#ifdef AEP_SUPPORTED
256,
256,
#endif
256,
256,
256,
256,
0
};

static const char *latency_tasks_str[BASIC_OPS_TASK_COUNT] = {
    "load-fence-64",
    "ntload-fence-64",
    "store-fence-64",
    "store-clflush-64",
#ifdef AEP_SUPPORTED
    "store-clwb-64",
    "store-clflushopt-64",
#endif
    "nstore-fence-64",
    "store-fence-movq-64",
    "store-clflush-movq-64",
    "load-fence-movq-64",

    "load-fence-128",
    "ntload-fence-128",
    "store-fence-128",
    "store-clflush-128",
#ifdef AEP_SUPPORTED
    "store-clwb-128",
    "store-clflushopt-128",
#endif
    "nstore-fence-128",
    "store-fence-movq-128",
    "store-clflush-movq-128",
    "load-fence-movq-128",

    "load-fence-256",
    "ntload-fence-256",
    "store-fence-256",
    "store-clflush-256",
#ifdef AEP_SUPPORTED
    "store-clwb-256",
    "store-clflushopt-256",
#endif
    "nstore-fence-256",
    "store-fence-movq-256",
    "store-clflush-movq-256",
    "load-fence-movq-256",
    "baseline"
};


uint64_t (*bench_func[BASIC_OPS_TASK_COUNT])(char *) = {
    &load_64byte_fence,
    &load_64byte_fence_nt,
    &store_64byte_fence,
    &store_64byte_clflush,
#ifdef AEP_SUPPORTED
    &store_64byte_clwb,
    &store_64byte_clflushopt,
#endif
    &nstore_64byte_fence,
    &store_64byte_fence_movq,
    &store_64byte_clflush_movq,
    &load_64byte_fence_movq,

    &load_128byte_fence,
    &load_128byte_fence_nt,
    &store_128byte_fence,
    &store_128byte_clflush,
#ifdef AEP_SUPPORTED
    &store_128byte_clwb,
    &store_128byte_clflushopt,
#endif
    &nstore_128byte_fence,
    &store_128byte_fence_movq,
    &store_128byte_clflush_movq,
    &load_128byte_fence_movq,


    &load_256byte_fence,
    &load_256byte_fence_nt,
    &store_256byte_fence,
    &store_256byte_clflush,
#ifdef AEP_SUPPORTED
    &store_256byte_clwb,
    &store_256byte_clflushopt,
#endif
    &nstore_256byte_fence,
    &store_256byte_fence_movq,
    &store_256byte_clflush_movq,
    &load_256byte_fence_movq,
    &baseline
};

/*
 * 256-byte benchmarks
 */
uint64_t repeat_256byte_ntstore(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_VALUE
        LOAD_ADDR
        TIMING_BEG
        "vmovntpd %%ymm0, 0*32(%%rsi) \n"
        "vmovntpd %%ymm0, 1*32(%%rsi) \n"
        "vmovntpd %%ymm0, 2*32(%%rsi) \n"
        "vmovntpd %%ymm0, 3*32(%%rsi) \n"
        "vmovntpd %%ymm0, 4*32(%%rsi) \n"
        "vmovntpd %%ymm0, 5*32(%%rsi) \n"
        "vmovntpd %%ymm0, 6*32(%%rsi) \n"
        "vmovntpd %%ymm0, 7*32(%%rsi) \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}

/*
 * 256-byte benchmarks
 */
uint64_t repeat_256byte_load(char *addr) {
    uint64_t t1 = 0, t2 = 0;
    uint64_t value = 0xC0FFEEEEBABE0000;
    KERNEL_BEGIN
    asm volatile(LOAD_VALUE
        LOAD_ADDR
        TIMING_BEG
        "vmovntdqa 0*32(%%rsi), %%ymm0 \n"
        "vmovntdqa 1*32(%%rsi), %%ymm1 \n"
        "vmovntdqa 2*32(%%rsi), %%ymm2 \n"
        "vmovntdqa 3*32(%%rsi), %%ymm3 \n"
        "vmovntdqa 4*32(%%rsi), %%ymm4 \n"
        "vmovntdqa 5*32(%%rsi), %%ymm5 \n"
        "vmovntdqa 6*32(%%rsi), %%ymm6 \n"
        "vmovntdqa 7*32(%%rsi), %%ymm7 \n"
        TIMING_END
        : [t1] "=r" (t1), [t2] "=r" (t2)
        : [memarea] "r" (addr), [value] "m" (value)
        : REGISTERS);
    KERNEL_END
    return t2 - t1;
}


#define SIZEBTNT_64_AVX512		\
				"vmovntdq  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"add $0x40, %%r10 \n"

#define SIZEBTNT_128_AVX512		\
				"vmovntdq  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"add $0x80, %%r10 \n"

#define SIZEBTNT_256_AVX512		\
				"vmovntdq  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"add $0x100, %%r10 \n"

#define SIZEBTNT_512_AVX512		\
				"vmovntdq  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"add $0x200, %%r10 \n"

#define SIZEBTNT_1024_AVX512	\
				"vmovntdq  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x200(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x240(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x280(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x2c0(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x300(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x340(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x380(%%r9, %%r10) \n" \
				"vmovntdq  %%zmm0,  0x3c0(%%r9, %%r10) \n" \
				"add $0x400, %%r10 \n"

#define SIZEBTSTFLUSH_64_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"clwb  0x0(%%r9, %%r10) \n" \
				"add $0x40, %%r10 \n"

#define SIZEBTSTFLUSH_128_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"clwb  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"clwb  0x40(%%r9, %%r10) \n" \
				"add $0x80, %%r10 \n"

#define SIZEBTSTFLUSH_256_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"clwb  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"clwb  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"clwb  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"clwb  0xc0(%%r9, %%r10) \n" \
				"add $0x100, %%r10 \n"

#define SIZEBTSTFLUSH_512_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"clwb  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"clwb  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"clwb  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"clwb  0xc0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"clwb  0x100(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"clwb  0x140(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"clwb  0x180(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"clwb  0x1c0(%%r9, %%r10) \n" \
				"add $0x200, %%r10 \n"

#define SIZEBTSTFLUSH_1024_AVX512	\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"clwb  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"clwb  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"clwb  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"clwb  0xc0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"clwb  0x100(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"clwb  0x140(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"clwb  0x180(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"clwb  0x1c0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x200(%%r9, %%r10) \n" \
				"clwb  0x200(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x240(%%r9, %%r10) \n" \
				"clwb  0x240(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x280(%%r9, %%r10) \n" \
				"clwb  0x280(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x2c0(%%r9, %%r10) \n" \
				"clwb  0x2c0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x300(%%r9, %%r10) \n" \
				"clwb  0x300(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x340(%%r9, %%r10) \n" \
				"clwb  0x340(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x380(%%r9, %%r10) \n" \
				"clwb  0x380(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x3c0(%%r9, %%r10) \n" \
				"clwb  0x3c0(%%r9, %%r10) \n" \
				"add $0x400, %%r10 \n"

#define SIZEBTST_64_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"add $0x40, %%r10 \n"

#define SIZEBTST_128_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"add $0x80, %%r10 \n"

#define SIZEBTST_256_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"add $0x100, %%r10 \n"

#define SIZEBTST_512_AVX512		\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"add $0x200, %%r10 \n"

#define SIZEBTST_1024_AVX512	\
				"vmovdqa64  %%zmm0,  0x0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x40(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x80(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0xc0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x100(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x140(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x180(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x1c0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x200(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x240(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x280(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x2c0(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x300(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x340(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x380(%%r9, %%r10) \n" \
				"vmovdqa64  %%zmm0,  0x3c0(%%r9, %%r10) \n" \
				"add $0x400, %%r10 \n"

#define SIZEBTLD_64_AVX512	\
				"vmovntdqa 0x0(%%r9, %%r10), %%zmm0 \n" \
				"add $0x40, %%r10 \n"

#define SIZEBTLD_128_AVX512	\
				"vmovntdqa  0x0(%%r9, %%r10), %%zmm0 \n" \
				"vmovntdqa  0x40(%%r9, %%r10), %%zmm1 \n" \
				"add $0x80, %%r10 \n"

#define SIZEBTLD_256_AVX512	\
				"vmovntdqa  0x0(%%r9, %%r10), %%zmm0 \n" \
				"vmovntdqa  0x40(%%r9, %%r10), %%zmm1 \n" \
				"vmovntdqa  0x80(%%r9, %%r10), %%zmm2 \n" \
				"vmovntdqa  0xc0(%%r9, %%r10), %%zmm3 \n" \
				"add $0x100, %%r10 \n"

#define SIZEBTLD_512_AVX512	\
				"vmovntdqa  0x0(%%r9, %%r10), %%zmm0 \n" \
				"vmovntdqa  0x40(%%r9, %%r10), %%zmm1 \n" \
				"vmovntdqa  0x80(%%r9, %%r10), %%zmm2 \n" \
				"vmovntdqa  0xc0(%%r9, %%r10), %%zmm3 \n" \
				"vmovntdqa  0x100(%%r9, %%r10), %%zmm4 \n" \
				"vmovntdqa  0x140(%%r9, %%r10), %%zmm5 \n" \
				"vmovntdqa  0x180(%%r9, %%r10), %%zmm6 \n" \
				"vmovntdqa  0x1c0(%%r9, %%r10), %%zmm7 \n" \
				"add $0x200, %%r10 \n"

#define SIZEBTLD_1024_AVX512	\
				"vmovntdqa  0x0(%%r9, %%r10), %%zmm0 \n" \
				"vmovntdqa  0x40(%%r9, %%r10), %%zmm1 \n" \
				"vmovntdqa  0x80(%%r9, %%r10), %%zmm2 \n" \
				"vmovntdqa  0xc0(%%r9, %%r10), %%zmm3 \n" \
				"vmovntdqa  0x100(%%r9, %%r10), %%zmm4 \n" \
				"vmovntdqa  0x140(%%r9, %%r10), %%zmm5 \n" \
				"vmovntdqa  0x180(%%r9, %%r10), %%zmm6 \n" \
				"vmovntdqa  0x1c0(%%r9, %%r10), %%zmm7 \n" \
				"vmovntdqa  0x200(%%r9, %%r10), %%zmm8 \n" \
				"vmovntdqa  0x240(%%r9, %%r10), %%zmm9 \n" \
				"vmovntdqa  0x280(%%r9, %%r10), %%zmm10 \n" \
				"vmovntdqa  0x2c0(%%r9, %%r10), %%zmm11 \n" \
				"vmovntdqa  0x300(%%r9, %%r10), %%zmm12 \n" \
				"vmovntdqa  0x340(%%r9, %%r10), %%zmm13 \n" \
				"vmovntdqa  0x380(%%r9, %%r10), %%zmm14 \n" \
				"vmovntdqa  0x3c0(%%r9, %%r10), %%zmm15 \n" \
				"add $0x400, %%r10 \n"

#define SIZEBT_NT_64	"movnti %[random], 0x0(%%r9, %%r10) \n" \
			"movnti %[random], 0x8(%%r9, %%r10) \n" \
			"movnti %[random], 0x10(%%r9, %%r10) \n" \
			"movnti %[random], 0x18(%%r9, %%r10) \n" \
			"movnti %[random], 0x20(%%r9, %%r10) \n" \
			"movnti %[random], 0x28(%%r9, %%r10) \n" \
			"movnti %[random], 0x30(%%r9, %%r10) \n" \
			"movnti %[random], 0x38(%%r9, %%r10) \n" \
			"add $0x40, %%r10 \n"

#define SIZEBT_LOAD_64	"mov 0x0(%%r9, %%r10),  %%r13  \n" \
			"mov 0x8(%%r9, %%r10),  %%r13  \n" \
			"mov 0x10(%%r9, %%r10), %%r13  \n" \
			"mov 0x18(%%r9, %%r10), %%r13  \n" \
			"mov 0x20(%%r9, %%r10), %%r13  \n" \
			"mov 0x28(%%r9, %%r10), %%r13  \n" \
			"mov 0x30(%%r9, %%r10), %%r13  \n" \
			"mov 0x38(%%r9, %%r10), %%r13  \n" \


#define REP2(S)        S ;        S
#define REP4(S)   REP2(S);   REP2(S)
#define REP8(S)   REP4(S);   REP4(S) 
#define REP16(S)  REP8(S);   REP8(S) 
#define REP32(S)  REP16(S);  REP16(S)
#define REP64(S)  REP32(S);  REP32(S)
#define REP128(S) REP64(S);  REP64(S)
#define REP256(S) REP128(S); REP128(S)
#define REP512(S) REP256(S); REP256(S)

#define KERNEL1(a,b,c)   ((a) = (a)*(b))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) +c)


#define SIZEBTNT_MACRO SIZEBTNT_512_AVX512
#define SIZEBTST_MACRO SIZEBTST_512_AVX512
#define SIZEBTLD_MACRO SIZEBTLD_512_AVX512
#define SIZEBTSTFLUSH_MACRO SIZEBTSTFLUSH_512_AVX512

//#define SIZEBTST_FENCE	"mfence \n"
//#define SIZEBTLD_FENCE	"mfence \n"
#define SIZEBTST_FENCE	""
#define SIZEBTLD_FENCE	""

#define CACHEFENCE_FENCE	"sfence \n"

  #define KERNEL_BEGIN do { } while (0);
  #define KERNEL_END do { } while (0);

#define SIZEBTNT_MACRO SIZEBTNT_512_AVX512
#define SIZEBTST_MACRO SIZEBTST_512_AVX512
#define SIZEBTLD_MACRO SIZEBTLD_512_AVX512
#define SIZEBTSTFLUSH_MACRO SIZEBTSTFLUSH_512_AVX512

//#define SIZEBTST_FENCE	"mfence \n"
//#define SIZEBTLD_FENCE	"mfence \n"
#define SIZEBTST_FENCE	""
#define SIZEBTLD_FENCE	""

#define CACHEFENCE_FENCE	"sfence \n"
//#define CACHEFENCE_FENCE	"mfence \n"


#define RandLFSR64 "mov    (%[random]), %%r9 \n"  \
				   "mov    %%r9, %%r12 \n"        \
				   "shr    %%r9 \n"               \
				   "and    $0x1, %%r12d \n"       \
				   "neg    %%r12 \n"              \
				   "and    %%rcx, %%r12 \n"       \
				   "xor    %%r9, %%r12 \n"        \
				   "mov    %%r12, (%[random]) \n" \
				   "mov    %%r12, %%r8 \n"        \
				   "and    %[accessmask], %%r8 \n"

#define CACHELINE_BITS			(6)
#define CACHELINE_SIZE			(64)
#define LATENCY_OPS_COUNT		1048576L

/*
 * Content on ReportFS
 *
 * Task 1: For random latency tests
 * | OP_COUNT * 8    |
 * +-----------------+---------------+---------------+-----+---------------+
 * |   Random Pool   | Report Task 0 | Report Task 1 | ... | Report Task N |
 * +-----------------+---------------+---------------+-----+---------------+
 *
 * Task 2: Strided latency
 * | OP_COUNT * 8  |
 * +---------------+---------------+-----+---------------+
 * | Report Task 0 | Report Task 1 | ... | Report Task N |
 * +---------------+---------------+-----+---------------+
 *
 * Task 5:Overwrite latency
 * | 2 * OP_COUNT * 8 |
 * +--------------------+
 * | Per-access latency |
 * +--------------------+
 *
 *
 */

#define OPT_NOPARAM	1
#define OPT_INT		2
#define OPT_STRING	4

/* LATENCY_OPS_COUNT = Subtasks (seq/rand) */
#define BASIC_OP_POOL_BITS		30
#define BASIC_OP_POOL_SIZE		(1L << POOL_BITS)  /* Size of the test region */
#define BASIC_OP_POOL_PAGE_BIT		(BASIC_OP_POOL_BITS - PAGE_SHIFT)
#define BASIC_OP_POOL_PAGES		(1L << BASIC_OP_POOL_PAGE_BIT) /* # Pages */

#define BASIC_OP_POOL_LINE_BITS		(BASIC_OP_POOL_BITS - CACHELINE_BITS)
#define BASIC_OP_MASK		0x3FFFFFC0 /*0b1{24, POOL_LINE_BITS}0{6, CACHELINE_BITS} */

#define PRETHREAD_BIT			31 /* 2GB/Thread */
//#define PRETHREAD_BIT			33 /* 8GB/Thread */
#define GLOBAL_BIT			34 /* 16GB/Global */
//#define GLOBAL_BIT			38 /* 256GB/Global */
#define GLOBAL_BIT_NI			32 /* 4GB/DIMM, 24 GB with software interleaving */
//#define PRETHREAD_BIT			28 /* 256MB/Thread */

#define GLOBAL_WORKSET			(1L << GLOBAL_BIT)
#define PERTHREAD_WORKSET		(1L << PRETHREAD_BIT)
#define PERDIMMGROUP_WORKSET		(PERTHREAD_WORKSET * 6)

//#define PERTHREAD_MASK		0xFFFFFC0 //256M
#define PERTHREAD_MASK			0x7FFFFFC0 //2G

#define PERTHREAD_CHECKSTOP		(1L << 25) /* yield after wrtting 32MB */

#define MB			0x100000
#define GB			0x40000000

#define DIMM_SIZE				(256UL * GB)
#define GLOBAL_WORKSET_NI		(1UL << GLOBAL_BIT_NI)


#define LFS_THREAD_MAX		48
#define LFS_ACCESS_MAX_BITS	22
#define LFS_ACCESS_MAX		(1 << LFS_ACCESS_MAX_BITS)  // 512KB * 4096 (rng_array) = 2G
#define LFS_RUNTIME_MAX		600
#define LFS_DELAY_MAX		1000000

#define LFS_PERMRAND_ENTRIES		0x1000
#define LFS_PERMRAND_SIZE			LFS_PERMRAND_ENTRIES * 4
#define LFS_PERMRAND_SIZE_IMM	"$0x4000,"



#define LAT_ASSERT(x)							\
	if (!(x)) {							\
		dump_stack();						\
		printk(KERN_WARNING "assertion failed %s:%d: %s\n",	\
	               __FILE__, __LINE__, #x);				\
	}


#define rdtscp(low,high,aux) \
     __asm__ __volatile__ (".byte 0x0f,0x01,0xf9" : "=a" (low), "=d" (high), "=c" (aux))


void stride_load(char *start_addr, long size, long skip, long delay, long count)
{
		KERNEL_BEGIN
		asm volatile (
		"xor %%r8, %%r8 \n"						/* r8: access offset */
		"xor %%r11, %%r11 \n"					/* r11: counter */

// 1
"LOOP_STRIDELOAD_OUTER_%=: \n"						/* outer (counter) loop */
		"lea (%[start_addr], %%r8), %%r9 \n"	/* r9: access loc */
		"xor %%r10, %%r10 \n"					/* r10: accessed size */
"LOOP_STRIDELOAD_INNER_%=: \n"						/* inner (access) loop, unroll 8 times */
		SIZEBTLD_64_AVX512						/* Access: uses r10[size_accessed], r9 */
		"cmp %[accesssize], %%r10 \n"
		"jl LOOP_STRIDELOAD_INNER_%= \n"
		SIZEBTLD_FENCE

		"xor %%r10, %%r10 \n"
"LOOP_STRIDELOAD_DELAY_%=: \n"						/* delay <delay> cycles */
		"inc %%r10 \n"
		"cmp %[delay], %%r10 \n"
		"jl LOOP_STRIDELOAD_DELAY_%= \n"

		"add %[skip], %%r8 \n"
		"inc %%r11 \n"
		"cmp %[count], %%r11 \n"

		"jl LOOP_STRIDELOAD_OUTER_%= \n"

		:: [start_addr]"r"(start_addr), [accesssize]"r"(size), [count]"r"(count), [skip]"r"(skip), [delay]"r"(delay):
			"%r11", "%r10", "%r9", "%r8");
		KERNEL_END
}


void kernel(uint64_t nsize,
            uint64_t ntrials,
            double* __restrict__ A,
            int* bytes_per_elem,
            int* mem_accesses_per_elem)
{
  *bytes_per_elem        = sizeof(*A);
  *mem_accesses_per_elem = 2;

  double alpha = 0.5;
  uint64_t i, j;
  for (j = 0; j < ntrials; ++j) {
  for (i = 0; i < nsize; ++i) {
      double beta = 0.8;
      KERNEL2(beta,A[i],alpha);
      A[i] = beta;
    }
    alpha = alpha * (1 - 1e-8);
  }
}
double getTime()
{
		double time;
		time = omp_get_wtime();
		return time;
}

void stride_nt(char *start_addr, long size, long skip, long delay, long count)
{
	//KERNEL_BEGIN
	asm volatile (
		"xor %%r8, %%r8 \n"						/* r8: access offset */
		"xor %%r11, %%r11 \n"					/* r11: counter */
		"movq %[start_addr], %%xmm0 \n"			/* zmm0: read/write register */
// 1
"LOOP_STRIDENT_OUTER_%=: \n"						/* outer (counter) loop */
		"lea (%[start_addr], %%r8), %%r9 \n"	/* r9: access loc */
		"xor %%r10, %%r10 \n"					/* r10: accessed size */
"LOOP_STRIDENT_INNER_%=: \n"						/* inner (access) loop, unroll 8 times */
		SIZEBTNT_64_AVX512							/* Access: uses r10[size_accessed], r9 */
		"cmp %[accesssize], %%r10 \n"
		"jl LOOP_STRIDENT_INNER_%= \n"
		SIZEBTLD_FENCE

		"xor %%r10, %%r10 \n"
//"LOOP_STRIDENT_DELAY: \n"						/* delay <delay> cycles */
//		"inc %%r10 \n"
//		"cmp %[delay], %%r10 \n"
//		"jl LOOP_STRIDENT_DELAY \n"
//
		"add %[skip], %%r8 \n"
		"inc %%r11 \n"
		"cmp %[count], %%r11 \n"

		"jl LOOP_STRIDENT_OUTER_%= \n"

		:: [start_addr]"r"(start_addr), [accesssize]"r"(size), [count]"r"(count), [skip]"r"(skip), [delay]"r"(delay):
			"%r11", "%r10", "%r9", "%r8");
	//KERNEL_END
}
void stride_store(char *start_addr, long size, long skip, long delay, long count)
{
	KERNEL_BEGIN
	asm volatile (
		"xor %%r8, %%r8 \n"						/* r8: access offset */
		"xor %%r11, %%r11 \n"					/* r11: counter */
		"movq %[start_addr], %%xmm0 \n"			/* zmm0: read/write register */
// 1
"LOOP_STRIDEST_OUTER: \n"						/* outer (counter) loop */
		"lea (%[start_addr], %%r8), %%r9 \n"	/* r9: access loc */
		"xor %%r10, %%r10 \n"					/* r10: accessed size */
"LOOP_STRIDEST_INNER: \n"						/* inner (access) loop, unroll 8 times */
		SIZEBTST_64_AVX512						/* Access: uses r10[size_accessed], r9 */
		"cmp %[accesssize], %%r10 \n"
		"jl LOOP_STRIDEST_INNER \n"
		SIZEBTST_FENCE

		"xor %%r10, %%r10 \n"
"LOOP_STRIDEST_DELAY: \n"						/* delay <delay> cycles */
		"inc %%r10 \n"
		"cmp %[delay], %%r10 \n"
		"jl LOOP_STRIDEST_DELAY \n"

		"add %[skip], %%r8 \n"
		"inc %%r11 \n"
		"cmp %[count], %%r11 \n"

		"jl LOOP_STRIDEST_OUTER \n"

		:: [start_addr]"r"(start_addr), [accesssize]"r"(size), [count]"r"(count), [skip]"r"(skip), [delay]"r"(delay):
			"%r11", "%r10", "%r9", "%r8");
	KERNEL_END
}


void stride_storeclwb(char *start_addr, long size, long skip, long delay, long count)
{
	KERNEL_BEGIN
	asm volatile (
		"xor %%r8, %%r8 \n"						/* r8: access offset */
		"xor %%r11, %%r11 \n"					/* r11: counter */
		"movq %[start_addr], %%xmm0 \n"			/* zmm0: read/write register */
// 1
"LOOP_STRIDESTFLUSH_OUTER: \n"						/* outer (counter) loop */
		"lea (%[start_addr], %%r8), %%r9 \n"	/* r9: access loc */
		"xor %%r10, %%r10 \n"					/* r10: accessed size */
"LOOP_STRIDESTFLUSH_INNER: \n"						/* inner (access) loop, unroll 8 times */
		SIZEBTSTFLUSH_64_AVX512						/* Access: uses r10[size_accessed], r9 */
		"cmp %[accesssize], %%r10 \n"
		"jl LOOP_STRIDESTFLUSH_INNER \n"
		SIZEBTST_FENCE

		"xor %%r10, %%r10 \n"
"LOOP_STRIDESTFLUSH_DELAY: \n"						/* delay <delay> cycles */
		"inc %%r10 \n"
		"cmp %[delay], %%r10 \n"
		"jl LOOP_STRIDESTFLUSH_DELAY \n"

		"add %[skip], %%r8 \n"
		"inc %%r11 \n"
		"cmp %[count], %%r11 \n"

		"jl LOOP_STRIDESTFLUSH_OUTER \n"

		:: [start_addr]"r"(start_addr), [accesssize]"r"(size), [count]"r"(count), [skip]"r"(skip), [delay]"r"(delay):
			"%r11", "%r10", "%r9", "%r8");
	KERNEL_END
}


int main(int argc, char *argv[]) {

		int rank = 0;
		int nprocs = 1;
		int nthreads = 1;
		int id = 0;

		uint64_t TSIZE = 1<<30;
        	TSIZE *=2;
		uint64_t PSIZE = TSIZE / nprocs;
		
		                int fd;
                if ( (fd = open("/mnt/daxtest/f", O_RDWR, 0666)) < 0){
                        printf("open file wrong!\n");
                        exit(1);
                }
                void * start;
                //
                if ((start=mmap(NULL, TSIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd,0))== MAP_FAILED)
                {
                        printf("mmap error!\n");
                        exit(1);
                }
                //double * buf = (double *)malloc(TSIZE);//start;
                //double * buf = (double *)start;
		char * buf = (char *) start;
		//char * buf = (char *)aligned_alloc(64,PSIZE);
		//char * buf = (char *)malloc(PSIZE);

		if (buf == NULL) {
				fprintf(stderr, "Out of memory!\n");
				return -1;
		}
#pragma omp parallel private(id) num_threads(4)
//#pragma omp parallel private(id) 
		{
				id = omp_get_thread_num();
				nthreads = omp_get_num_threads();

				uint64_t nsize = PSIZE / nthreads;
				nsize = nsize & (~(64-1));
				nsize = nsize / sizeof(double)/8;
				uint64_t nid =  nsize * id;

				// initialize small chunck of buffer within each thread
				//initialize(nsize, &buf[nid], 1.0);


				double startTime, endTime;
				uint64_t n,nNew;
				uint64_t t;
				int bytes_per_elem;
				int mem_accesses_per_elem;

				int it = 0;
				n = nsize; t = 1;
				while (it < 10) { // working set - nsize
				printf("start %d\n", it);	
				#pragma omp barrier

								if ((id == 0) && (rank==0)) {
										startTime = getTime();
								}
								// C-code
								//printf("before start\n");
								//stride_load(&buf[nid],64,64,0,nsize);
								stride_store(&buf[nid],64,64,0,nsize);
								//stride_storeclwb(&buf[nid],64,64,0,nsize);
								//stride_nt(&buf[nid],64,64,0,nsize);
								//printf("before start\n");
								//kernel(n, t, &buf[nid], &bytes_per_elem, &mem_accesses_per_elem);
								//msync(&buf[nid], n*sizeof(double), MS_SYNC);
				#pragma omp barrier

								if ((id == 0) && (rank == 0)) {
										endTime = getTime();
										double seconds = (double)(endTime - startTime)*1000000;
										uint64_t working_set_size = n * nthreads * nprocs;
										uint64_t total_bytes = t * working_set_size * 64 * 1;// * mem_accesses_per_elem;
										uint64_t total_flops = t * working_set_size * ERT_FLOP;
										// nsize; trials; microseconds; bytes; single thread bandwidth; total bandwidth
										//printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n",
										//       working_set_size * bytes_per_elem,
										//       t,
										//       seconds,
										//       total_bytes,
										//       total_flops);
                                        printf("BW: %15.3lf Total data: %15.3lfG \n",total_bytes*1.0/seconds/1.024/1.024/1024, total_bytes*1.0/1024/1024/1024);
								} // print
					it++; 
				} // working set - nsize
		} // parallel region
		free(buf);
		//munmap(buf, TSIZE);
		//close(fd);
		printf("\n");
		printf("META_DATA\n");
		printf("FLOPS          %d\n", ERT_FLOP);
		printf("OPENMP_THREADS %d\n", nthreads);

		return 0;
}
