// Storage precision never changes accumulator precision.
#ifndef PRECISION_MASK
#define PRECISION_MASK 0
#endif
#if PRECISION_MASK
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#if PRECISION_MASK & 1
#define STORAGE_0 float16_t
#else
#define STORAGE_0 float
#endif
#if PRECISION_MASK & 2
#define STORAGE_1 float16_t
#else
#define STORAGE_1 float
#endif
#if PRECISION_MASK & 4
#define STORAGE_2 float16_t
#else
#define STORAGE_2 float
#endif
#if PRECISION_MASK & 8
#define STORAGE_3 float16_t
#else
#define STORAGE_3 float
#endif
#if PRECISION_MASK & 16
#define STORAGE_4 float16_t
#else
#define STORAGE_4 float
#endif
#if PRECISION_MASK & 32
#define STORAGE_5 float16_t
#else
#define STORAGE_5 float
#endif
