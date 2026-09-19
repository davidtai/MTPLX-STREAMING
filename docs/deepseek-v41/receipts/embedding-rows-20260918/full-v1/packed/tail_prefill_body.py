def _forward_layer_major(self, input_ids, cache, chunk, *,
                         return_main_hidden: bool = False):
    """Layer-major chunked prefill (K16): iterate every layer over all chunks
    before the next layer, so each layer's routed-expert bank is streamed
    ONCE across the whole prompt instead of once per chunk.

    Correctness vs the chunk-major driver (:meth:`_forward_span` per span):

    * **Attention stays causal + per chunk.** Within a layer the chunks run in
      order 0..C-1; chunk ``c`` appends its post-RoPE KV to the same
      append-only layer store and reads the accumulated window, so it attends
      over chunks ``< c`` exactly as one-shot -- and the ``[chunk, H, T]`` score
      transient that motivated W20 stays bounded to one chunk (never
      concatenated).  Each chunk's attention half is evaluated before the next
      chunk builds its graph, so only one score is live at a time.
    * **The MoE reads the bank once.** After a layer's C attention halves, the
      chunks' routed-expert inputs are concatenated and fed to ``mlp`` in one
      ``switch_mlp`` call (row-capped, below), so ``partition_route_waves``
      gathers each of the layer's experts exactly once for the whole prompt.
    * **Hyper-Connection state is resident per chunk.** Every chunk keeps its
      own ``[b, chunk, hc_mult, hidden]`` stream and ``pre_mix`` across the
      whole layer loop (all C together are ``hidden * hc_mult * s * bf16`` --
      0.67 GB at 16 K, well under 1 GB); the ffn ``carry`` is transient within
      a layer.  A per-chunk :class:`SharedAttentionRuntime` threads each
      chunk's compressed-KV / index selection down the stack exactly as its
      span would.
    * **Engram + DSpark unchanged in order.** The engram history is advanced
      once per chunk in position order up front (identical ``_buf``/``_len`` to
      chunk-major) and each chunk's row ids are replayed to the engram hook via
      a per-chunk view, so layers 1/14 write the same residual for the same
      rows.  ``main_hidden`` captures the target-layer input per chunk and
      concatenates in position order, so the DSpark draft seed spans the whole
      prompt (its ``[:, -1:, :]`` slice is still the final prompt token)."""
    _stime.set_schedule("layer_major")
    b, s = input_ids.shape
    capture_start = max(0, s - _tail_keep_rows)
    # W107 (review LOW-1): fail an over-cap prefill BEFORE any lane is written.
    _admit = getattr(cache, "assert_can_admit", None)
    if callable(_admit):
        _admit(s)
    offset0 = int(cache.offset)
    spans = [(start, min(start + chunk, s)) for start in range(0, s, chunk)]
    n_chunks = len(spans)

    engram_state = getattr(cache, "engram_state", None)
    want_main = return_main_hidden and bool(self._mtp_target_layer_ids)

    # Per-chunk resident state, built once.  The engram history is advanced in
    # position order here (so `_buf`/`_len` end identical to chunk-major) and
    # each chunk's returned row ids are captured for the hook replay below --
    # the shared `_current` only holds the last advance, so we never read it.
    hs: List[mx.array] = []
    pre_mixes: List[mx.array] = []
    positions_all: List[mx.array] = []
    engram_currents: List[Optional[np.ndarray]] = []
    shareds = [cache.new_shared_runtime() for _ in range(n_chunks)]
    main_hiddens: List[List[mx.array]] = [[] for _ in range(n_chunks)]
    for c, (start, end) in enumerate(spans):
        ids_c = input_ids[:, start:end]
        n_c = end - start
        positions_all.append(mx.arange(offset0 + start, offset0 + end))
        # W47: embed + engram.advance per chunk (tagged by chunk index), so the
        # layer-major flat stages match chunk-major.  ``stage``/``chunk`` are
        # no-ops off / decode; this loop is layer-major-only regardless.
        with _stime.chunk(c):
            with _stime.stage("embed") as _st:
                h_c = self.embed_tokens(ids_c)
                h_c = mx.broadcast_to(
                    h_c[:, :, None, :], (b, n_c, self.hc_mult, h_c.shape[-1])
                )
                _st.add(h_c)
            hs.append(h_c)
            pre_mixes.append(
                mx.concatenate(
                    [mx.ones((b, n_c, 1)), mx.zeros((b, n_c, self.hc_mult - 1))],
                    axis=-1,
                ).astype(mx.float32)
            )
            if engram_state is not None:
                with _stime.stage("engram.advance"):
                    engram_currents.append(engram_state.advance(ids_c))
            else:
                engram_currents.append(None)

    row_cap = _derive_moe_row_cap(self.args, _prefill_moe_row_target_bytes())
    # Small verification calls can select this schedule with an explicit
    # tiny chunk. Keep their decode caches; large prefills need each dense
    # projection only until that layer's final chunk has completed.
    release_dense_projection = b * s > _DECODE_ATTN_KERNEL_MAX_ROWS

    for layer in self.layers:
        lc = cache.layers[layer.layer_id]
        is_target = want_main and layer.layer_id in self._mtp_target_layer_ids
        moe_inputs: List[mx.array] = []
        carries: List[tuple] = []
        for c, (start, end) in enumerate(spans):
            # W47: tag this (layer, chunk) attention half with the chunk index
            # so ``by_chunk`` accumulates attention/HC/engram per chunk across
            # every layer (the MoE is batched below, outside any chunk tag).
            with _stime.chunk(c):
                h_c = hs[c]
                if layer.engram_hook is not None and engram_state is not None:
                    h_c = layer.engram_hook(
                        h_c, input_ids[:, start:end],
                        _ChunkEngramView(engram_currents[c]),
                    )
                if is_target and end > capture_start:
                    main_hiddens[c].append(
                        mx.mean(h_c[:, max(0, capture_start - start):].astype(mx.float32), axis=2).astype(h_c.dtype)
                    )
                moe_in_c, carry_c, ffn_pre_c = layer.attn_and_moe_input(
                    h_c, pre_mixes[c], positions_all[c], lc, shareds[c]
                )
                moe_inputs.append(moe_in_c)
                carries.append(carry_c)
                pre_mixes[c] = ffn_pre_c
                # Free this chunk's attention score before the next chunk's
                # graph is built (only one [chunk, H, T] transient live at once).
                # Materialize captured means here too, so their lazy graphs
                # do not retain earlier layers' full Hyper-Connection states.
                self._eval_layer_transients(
                    lc, moe_in_c, ffn_pre_c, main_hiddens[c]
                )

        # One routed-expert call per layer over every chunk's rows -> the bank
        # is streamed once.  Split only if the row cap (routed-output transient
        # budget) would be exceeded; at 16 K the whole prompt is one call.
        moe_outputs = self._layer_major_moe(layer, moe_inputs, spans, row_cap)
        for c in range(n_chunks):
            with _stime.stage("hc.combine") as _st:
                hs[c] = _PREFILL_HC_POST(moe_outputs[c], *carries[c])
                _st.add(hs[c])
        mx.eval(hs)
        if release_dense_projection:
            # This existing fence has consumed the attention and FFN
            # carries. Preserve reuse across chunks, then release before
            # the next layer instead of retaining 40 fp32 weight copies.
            layer.attn._wo_a_dense_cache = None

    cache.advance(s)

    outputs: List[mx.array] = []
    main_parts: List[Optional[mx.array]] = []
    for c in range(n_chunks):
        with _stime.chunk(c), _stime.stage("final_norm") as _st:
            h = mx.sum(
                pre_mixes[c][..., None] * hs[c].astype(mx.float32), axis=2
            ).astype(hs[c].dtype)
            out_c = _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)
            _st.add(out_c)
        outputs.append(out_c)
        if main_hiddens[c]:
            main_parts.append(mx.concatenate(main_hiddens[c], axis=-1))
    out = mx.concatenate(outputs, axis=1)
    if not return_main_hidden:
        return out
    main_hidden = (
        None
        if any(p is None for p in main_parts)
        else mx.concatenate(main_parts, axis=1)
    )
    return out, main_hidden
