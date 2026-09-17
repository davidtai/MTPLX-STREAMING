#include <mach/task_info.h>
#include <stddef.h>
#include <stdio.h>

_Static_assert(TASK_VM_INFO == 22, "task flavor");
_Static_assert(TASK_VM_INFO_REV1_COUNT == 38, "rev1 word count");
_Static_assert(offsetof(task_vm_info_data_t, resident_size) == 16, "resident offset");
_Static_assert(offsetof(task_vm_info_data_t, compressed) == 120, "compressed offset");
_Static_assert(offsetof(task_vm_info_data_t, phys_footprint) == 144, "footprint offset");

int main(void) {
    printf("{\"sdk_abi_matches\":true,\"flavor\":%d,\"rev1_words\":%u,\"rev1_bytes\":152}\n",
           TASK_VM_INFO, TASK_VM_INFO_REV1_COUNT);
    return 0;
}
