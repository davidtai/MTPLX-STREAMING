#include <mach/host_info.h>
#include <mach/vm_statistics.h>
#include <stddef.h>
#include <stdio.h>

_Static_assert(HOST_VM_INFO64 == 4, "host flavor");
_Static_assert(HOST_VM_INFO64_REV1_COUNT == 38, "rev1 word count");
_Static_assert(sizeof(natural_t) == 4, "counter width");
_Static_assert(offsetof(vm_statistics64_data_t, speculative_count) == 92, "speculative offset");
_Static_assert(offsetof(vm_statistics64_data_t, swapins) == 112, "swap offset");
_Static_assert(offsetof(vm_statistics64_data_t, compressor_page_count) == 128, "physical compressor offset");
_Static_assert(offsetof(vm_statistics64_data_t, total_uncompressed_pages_in_compressor) == 144, "logical compressor offset");

int main(void) {
    printf("{\"sdk_abi_matches\":true,\"flavor\":%d,\"rev1_words\":%u,\"rev1_bytes\":152}\n",
           HOST_VM_INFO64, HOST_VM_INFO64_REV1_COUNT);
    return 0;
}
