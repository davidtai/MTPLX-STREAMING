"""Exact native BF16 output weights, resident or expanded one call ahead."""
import mlx.core as mx
from attention_reader import load_attention_tensors
from fused_transpose import make_transpose


class OutputStore:
    def __init__(self, root, manifest, proof, *, resident):
        self.packed = []
        self.up = []
        self.dense = []
        self.buffers = [None, None]
        self.expand = make_transpose()
        kept = tuple(t for t in manifest.resident_tensors if t.tensor in proof['resident_names'])
        raw = load_attention_tensors(root, manifest, kept, mx=mx)
        for layer in range(40):
            prefix = f'layers.{layer}.attn.'
            w,s,bw,bs = (raw.pop(prefix+n) for n in
                ('wo_a.weight','wo_a.scales','wo_b.weight','wo_b.scales'))
            if (w.dtype != mx.uint32 or tuple(w.shape) != (8192,1024)
                    or s.dtype != mx.uint8 or tuple(s.shape) != (8192,128)
                    or bw.dtype != mx.uint32 or tuple(bw.shape) != (5120,2048)
                    or bs.dtype != mx.uint8 or tuple(bs.shape) != (5120,256)):
                raise RuntimeError('native MXFP8 projection geometry differs')
            self.up.append((bw,bs))
            if resident:
                value = mx.contiguous(mx.dequantize(w,s,None,group_size=32,bits=8,mode='mxfp8')
                    .astype(mx.bfloat16).reshape(8,1024,4096).swapaxes(1,2))
                mx.eval(value)
                self.dense.append(value)
            else:
                self.packed.append((w,s))
        if raw:
            raise RuntimeError('unconsumed projection tensors')

    def issue(self, step):
        # Called after source demand submission and its covering router eval.
        # Each expansion owns fresh output; previous consumers are never mutated.
        value = self.expand(*self.packed[step % 40])
        mx.async_eval(value)
        self.buffers[step % 2] = value

    def acquire(self, step):
        return self.buffers[step % 2]

    def resident_acquire(self, step):
        return self.dense[step % 40]

    def project(self, value, weight, step):
        grouped = value.reshape(6,8,4096).swapaxes(0,1)
        output = mx.matmul(grouped,weight).swapaxes(0,1).reshape(1,6,8192)
        bw,bs = self.up[step % 40]
        return mx.quantized_matmul(output,bw,bs,None,transpose=True,
                                  group_size=32,bits=8,mode='mxfp8')

    def close(self):
        self.buffers.clear()
        self.dense.clear()
        self.packed.clear()
        self.up.clear()
        self.expand = None
