import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
dest=ROOT/'candidate'
for name in ('stage.py','ar_body.txt','q8_admission.py'):
    shutil.copyfile(ROOT/name,dest/name)
spec=importlib.util.spec_from_file_location('q8_candidate_setup',dest/'stage.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source=module.source
def change(old,new):
    global source
    if source.count(old)!=1:
        raise RuntimeError('candidate substitution count: '+old[:100])
    source=source.replace(old,new)
change("args.decode_mode != 'ar' or not args.with_mtp", "args.decode_mode != 'dspark'")
change("args.host_overhead_gib != 2.625", "args.host_overhead_gib != 2")
change("REFERENCE_SOURCE_COMMIT = 'e589c1e4b17856f506d90d9fb2bbb5ce45711648'",
       f"REFERENCE_SOURCE_COMMIT = {module.c['source_commit']!r}")
change("if (reference_bounds.get('source_commit') != REFERENCE_SOURCE_COMMIT",
       "if (reference_bounds.get('source_commit') != REFERENCE_SOURCE_COMMIT\n"
       "    or not reference.get('reference_only')\n"
       "    or reference.get('fixed_q8_cache',{}).get('bits') != 8\n"
       "    or reference.get('fixed_q8_cache',{}).get('max_kv') != 17664\n"
       "    or reference.get('fixed_q8_cache',{}).get('max_append') != 953")
change("raise RuntimeError('AR reference is not a complete matched native-target run')",
       "raise RuntimeError('AR reference is not a complete matched Q8-target run')\n"
       "logit_storage = reference['reference_logit_storage']\n"
       "logit_path = reference_path.with_suffix('.ar-logits.f32')\n"
       "if (Path(logit_storage['path']).resolve() != logit_path.resolve()\n"
       "    or logit_storage['shape'] != [1024,129280] or logit_storage['dtype'] != 'float32'\n"
       "    or len(logit_storage['row_sha256']) != 1024 or logit_storage['row_bytes'] != 517120\n"
       "    or logit_path.stat().st_size != 529530880):\n"
       "    raise RuntimeError('Q8 reference logit inventory is incomplete')")
start=source.index('    def cached_ar_logits_row(**kw):')
end=source.index('    ab._ar_logits_row_at_index = cached_ar_logits_row',start)
source=source[:start]+'''    def cached_ar_logits_row(**kw):
        index=int(kw['index'])
        if list(kw['ar_tokens'])!=reference_ids or not 0<=index<1024:
            raise RuntimeError('Q8 diagnostic prefix/index differs from reference')
        fd=os.open(logit_path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            blob=os.pread(fd,517120,index*517120)
        finally:
            os.close(fd)
        if len(blob)!=517120 or hashlib.sha256(blob).hexdigest()!=logit_storage['row_sha256'][index]:
            raise RuntimeError('Q8 reference row digest differs')
        row=ab.np.frombuffer(blob,dtype=ab.np.float32)
        ar_logits_cache_report.update(index=index,source_commit=REFERENCE_SOURCE_COMMIT,
            reference_receipt_sha256=reference_provenance['receipt_sha256'],
            path=str(logit_path),row_sha256=logit_storage['row_sha256'][index],
            payload_bytes=517120,complete_row=True,kv_bits=8,reused=True)
        return row

'''+source[end:]
change("if receipt['dspark']['token_ids_sha256'] != CONTROL_OUTPUT_SHA256:",
       "if len(receipt['dspark']['token_ids']) != 1024:")
change("raise RuntimeError('candidate changed the full validated MTP token digest')",
       "raise RuntimeError('Q8 MTP did not produce the full 1024-token workload')")
ast.parse(source)
(dest/'run_full.py').write_text(source)
p=json.loads((dest/'packed/installation.json').read_text())
p['helper_sha256']['../run_full.py']=hashlib.sha256(source.encode()).hexdigest()
(dest/'packed/installation.json').write_text(json.dumps(p,indent=2)+'\n')
print('Staged candidate with fresh Q8 reference and complete-row divergence evidence:',
      hashlib.sha256(source.encode()).hexdigest())
