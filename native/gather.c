#include <stddef.h>
#include <stdint.h>

#ifndef SOURCE_SHA256
#error "Build through build_native_preprocess.sh to bind the binary to its source"
#endif

__attribute__((visibility("default")))
const char *mac_gather_build_id(void) { return SOURCE_SHA256; }

// Integer-only gather: no alternate interpolation, floating-point or color math.
// Bounds are checked before each dereference, including for malformed index maps.
__attribute__((visibility("default")))
int mac_gather_bytes(const uint8_t *source, size_t source_size,
                     const uint32_t *indices, size_t count,
                     uint8_t *destination, size_t destination_size) {
  if (!source || !indices || !destination || count > destination_size) return 1;
  for (size_t i = 0; i < count; ++i) {
    uint32_t offset = indices[i];
    if ((size_t)offset >= source_size) return 2;
    destination[i] = source[offset];
  }
  return 0;
}
