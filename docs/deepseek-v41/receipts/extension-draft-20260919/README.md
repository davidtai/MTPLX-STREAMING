# Smaller draft plus extension bank: admission refusal

Source `a15ba7c40b6e59c7e3c3fdae852112085bddf40e`. This composes the existing
80/40/24 native Q4 draft subset with the measured extension bank and predictable
projection schedule. Target arithmetic, native KV16, D5 plus lookup, and M<=8
verification remain unchanged. All runner callable ASTs match the prior winner.

The subset removes 733,224,960 payload bytes. Accounting adds a separately
itemized 268,435,456-byte background variation reserve following the preceding
full run's 107,409,300-byte estimate overrun. The aggregate non-MLX allowance is
1,707,208,704 bytes, including the unchanged 1,438,773,248-byte host reserve.
The physical limit remains exactly 110,000,000,000 bytes.

On the guarded attempt, fresh background was 11,560,878,080 bytes. The intended
minimum 112 rows/layer did not fit. The runner refused before model loading,
prefill, or decode; there is no full throughput, output, or peak-model result.
Shifting the archived CPU admission case to this exact baseline yields a
112-row physical bound of 110,567,052,536 bytes, 567,052,536 above the ceiling.
That reconstruction is a calculation, not a saved live admission measurement.
All three subset files were authenticated before the refusal.

Guard exit was 1. Sampled child footprint peaked at 49,496,568 bytes and machine
usage at 11,625,349,120 bytes, with no compressor growth. Qwen identity, health,
and warmup were restored before release at 15:34:50 UTC; an independent
15:36:27 UTC check found healthy idle Qwen, a free lock, and no owned child.
Do not weaken admission or repeat the unchanged full candidate to seek a lower
background. The retained best remains 13.8688167379 TPS; 20 TPS is unmet.

Sources and CPU checks are preserved here; immutable model and packed-scale
artifacts remain at their authenticated local paths. No optimization regression
suite was added for this refused attempt.
