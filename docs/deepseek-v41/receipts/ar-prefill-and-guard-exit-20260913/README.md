# AR prefill allocation and guard exit fixes

The benchmark's five AR prefill sites now request `logits_keep=1`: the main
pass, divergence replay, stage timing, prefill timing and synchronization
census. The 16K f32 vocabulary output falls from 8,472,494,080 B to 517,120 B.
This is output-allocation accounting, not a measured global peak reduction.
Single-token decode and multi-row DSpark verification are unchanged. The
Metal head's changed row geometry can alter rounding, so the tie-only numerical
gate remains necessary.

`Model.__call__` also stops requesting draft hidden captures when
`return_hidden=False`; it conditionally unpacks the backbone's existing
return contracts. DSpark and `hc_hidden()` still explicitly request captures.
Future DSpark bounds derived from a lower AR peak must separately include the
retained target HC inputs. Do not reuse the old assumption that AR already
holds them; the previous measured DSpark peak remains available as a control.

Four new CPU allocation cases failed before the fix; the capture-required case
already passed. All five pass afterward, together with eight selected existing
EOS/warm-repeat checks. Real MLX/MLX-LM imports were blocked throughout.

The guard failure was independently reproduced with a 64 MiB CPU child in four
trials: `ps` returned `?E`, exit status 0, while `kill(pid,0)` still saw the child.
Apple's [Mach state table](https://github.com/apple-oss-distributions/adv_cmds/blob/main/ps/tasks.c)
and [state printer](https://github.com/apple-oss-distributions/adv_cmds/blob/main/ps/print.c)
explain the unknown thread-state marker followed by the process-exiting flag.
The guard now recognizes that exact suffix grammar as **live**, keeps physical
and process memory/compressor checks running, and waits for the ordinary
gone/zombie transition. Bare `?`, malformed states and reader failures still
fail closed.

Two focused mocked-service regressions failed with the previous guard's exit 8:
preserve an owned child's exit 7, and abort an over-cap exiting-state child with
exit 6. Both now pass, along with the existing unreadable-live-state abort case.
No real service or GPU was used by these tests. Raw red/green logs and the
bounded native CPU reproduction are included.
