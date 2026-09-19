# Possible next experiment, unmeasured

The measured 110→111 native resize costs about45ms/layer. The retained complete
run also charges3.562s for its initial84→110 packed growth. Current full-v1 only
avoids a second resize; it still copies the initial bank once.

The packed kernels already group physical rows by backing bank. A separate
post-prefill extension bank could keep the original84 rows in place and add
the remaining rows without copying existing weights. This might reduce the
first growth time and remove its component-copy peak, but resident hits would
span two banks much more often than in the measured one-row overflow variant.
Extra gathers/kernel dispatches may erase the one-time saving.

If pursued, first run one bounded actual-layout comparison with the real saved
routes, exact packed storage and native arithmetic, charging allocation cost.
Do not extrapolate the one-row result to a large extension bank. Re-derive all
phase bounds before any full request; the native seed or steady state may then
be limiting. This is a hypothesis, not a promoted optimization or proven bound.
