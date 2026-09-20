/* DSV4.1 F37 — beam-256 trellis step in C (CPU, single-thread), called via ctypes.
 *
 * Replaces the numpy per-step top-256-of-2048 introselect (the ~3.3 ms/tile wall in
 * beam_encode_fast) with a bounded max-heap: the 256 surviving beams are held in a heap keyed by
 * (cost, candidate-index); each of the 2048 expansion candidates is compared once against the heap
 * root and dropped in O(1) if it is not better than the current 256th, so selection is no longer a
 * full 2048-element sort per step.
 *
 * Arithmetic is byte-identical to beam_encode_fast: float32 emission  cost = beam_cost +
 * (DEC2[w] - (2*t)*DEC[w])  with the SAME DEC / DEC2 tables passed in from Python.  The ONLY
 * behavioural difference is the tie rule: on EQUAL cost the lower candidate index wins
 * (deterministic), whereas numpy's argpartition tie-break is implementation-defined.  On real
 * weight data the accumulated float32 costs are distinct, so the surviving SET — and therefore the
 * backtracked code — is identical to F34's beam (verified against the stored artifact).
 *
 * Output is the OPEN-CHAIN new3 [N,256]; the caller applies the identical numpy seam-repair.
 *
 *  targets  [N,S]      cycle-order target weights (float32)
 *  DEC,DEC2 [65536]     flat codebook + its square (step 0 windows w = g*8 + j are contiguous)
 *  DECg2,DEC2g2 [8192*8] per-state contiguous layout: DECg2[s*8+r] = DEC[s + r*8192] (steps 1..S-1)
 *  out      [N,S]       open-chain new3 (uint8)
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct { float cost; int idx; int ns; unsigned char sym; unsigned char par; } Cand;

/* strict "greater than" by (cost, idx): equal cost -> larger idx is greater (so lower idx is kept) */
static inline int cand_gt(const Cand *a, const Cand *b) {
    if (a->cost > b->cost) return 1;
    if (a->cost < b->cost) return 0;
    return a->idx > b->idx;
}

static inline void sift_up(Cand *h, int i) {
    Cand c = h[i];
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (cand_gt(&c, &h[p])) { h[i] = h[p]; i = p; } else break;
    }
    h[i] = c;
}

/* max-heap sift-down of the root over a heap of size n */
static inline void sift_down(Cand *h, int n) {
    int i = 0;
    Cand c = h[0];
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        Cand mv = c;
        if (l < n && cand_gt(&h[l], &mv)) { m = l; mv = h[l]; }
        if (r < n && cand_gt(&h[r], &mv)) { m = r; mv = h[r]; }
        if (m == i) break;
        h[i] = h[m];
        i = m;
    }
    h[i] = c;
}

/* push a candidate into the bounded (size `beam`) max-heap that keeps the `beam` SMALLEST by
 * (cost, idx); *hn is the current size. */
static inline void heap_offer(Cand *h, int *hn, int beam, const Cand *cand) {
    if (*hn < beam) { h[*hn] = *cand; sift_up(h, *hn); (*hn)++; }
    else if (cand_gt(&h[0], cand)) { h[0] = *cand; sift_down(h, beam); }   /* cand < root -> replace */
}

int tcq_beam_batch(const float *targets, long N, int S, int beam,
                   const float *DEC, const float *DEC2,
                   const float *DECg2, const float *DEC2g2,
                   unsigned char *out) {
    if (beam <= 0 || S <= 0) return 1;
    Cand *heap = (Cand *) malloc(sizeof(Cand) * beam);
    int   *bstate = (int *)   malloc(sizeof(int)   * beam);
    float *bcost  = (float *) malloc(sizeof(float) * beam);
    unsigned char *par = (unsigned char *) malloc((size_t) S * beam);
    unsigned char *sym = (unsigned char *) malloc((size_t) S * beam);
    if (!heap || !bstate || !bcost || !par || !sym) return 2;

    for (long b = 0; b < N; b++) {
        const float *tg = targets + (size_t) b * S;
        unsigned char *ob = out + (size_t) b * S;

        /* ---- step 0: free-init collapse, window w = g*8 + j, keep top-`beam` next-states g ---- */
        float two_t = 2.0f * tg[0];
        int hn = 0;
        for (int g = 0; g < 8192; g++) {
            const float *d = DEC + (size_t) g * 8;      /* DEC[g*8 .. g*8+7] contiguous */
            const float *d2 = DEC2 + (size_t) g * 8;
            float best = INFINITY; int bestj = 0;
            for (int j = 0; j < 8; j++) {
                float e = d2[j] - two_t * d[j];
                if (e < best) { best = e; bestj = j; }  /* argmin: first (lowest j) on ties */
            }
            Cand cand = { best, g, g, (unsigned char) (((g << 3) | bestj) >> 13), 0 };
            heap_offer(heap, &hn, beam, &cand);
        }
        for (int k = 0; k < beam; k++) {
            bstate[k] = heap[k].ns; bcost[k] = heap[k].cost;
            par[k] = heap[k].par; sym[k] = heap[k].sym;
        }

        /* ---- steps 1..S-1: expand 256 beams x 8, keep top-`beam` ---- */
        for (int p = 1; p < S; p++) {
            two_t = 2.0f * tg[p];
            hn = 0;
            for (int i = 0; i < beam; i++) {
                int s = bstate[i];
                float bc = bcost[i];
                const float *d = DECg2 + (size_t) s * 8;    /* DEC[s + r*8192] for r=0..7 */
                const float *d2 = DEC2g2 + (size_t) s * 8;
                int ns_base = s >> 3;
                for (int r = 0; r < 8; r++) {
                    float e = d2[r] - two_t * d[r];
                    Cand cand = { bc + e, i * 8 + r, ns_base | (r << 10),
                                  (unsigned char) r, (unsigned char) i };
                    heap_offer(heap, &hn, beam, &cand);
                }
            }
            unsigned char *parp = par + (size_t) p * beam;
            unsigned char *symp = sym + (size_t) p * beam;
            for (int k = 0; k < beam; k++) {
                bstate[k] = heap[k].ns; bcost[k] = heap[k].cost;
                parp[k] = heap[k].par; symp[k] = heap[k].sym;
            }
        }

        /* ---- backtrack from the min-cost beam (lowest slot on tie) ---- */
        float mc = INFINITY; int slot = 0;
        for (int k = 0; k < beam; k++) if (bcost[k] < mc) { mc = bcost[k]; slot = k; }
        for (int p = S - 1; p >= 0; p--) {
            ob[p] = sym[(size_t) p * beam + slot];
            slot = par[(size_t) p * beam + slot];
        }
    }
    free(heap); free(bstate); free(bcost); free(par); free(sym);
    return 0;
}
