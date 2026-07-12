#pragma once
#include "rust/cxx.h"

// Expose the KV Cache memory quantization logic to Rust via cxx
void apply_kv_cache_quantization(rust::Str precision);
