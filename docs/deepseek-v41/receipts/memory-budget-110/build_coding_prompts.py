import hashlib, json, pathlib
from tokenizers import Tokenizer
from jinja2 import Environment
from mtplx.chat_encoding import render_deepseek_v41_prompt
root=pathlib.Path.cwd(); artifact=pathlib.Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
output=root/'docs/deepseek-v41/receipts/memory-budget-110'
context_path=root/'mtplx/benchmarks/prompts/qwen38_generation_context.py'
context=context_path.read_text(); digest=lambda b:hashlib.sha256(b).hexdigest()
assert digest(context.encode())=='c8ae2b1790c0300aa7c1421b55e7cd5d43c93461f7fba5d3a732fd34e156b4c4'
instruction=json.loads((root/'mtplx/benchmarks/prompts/qwen38_naturalistic_generation_patch.jsonl').read_text().splitlines()[0])
tokenizer=Tokenizer.from_file(str(artifact/'tokenizer.json'))
template=Environment().from_string((artifact/'chat_template.jinja').read_text())
encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
seed=20260829;lines=context.splitlines();offset=seed%len(lines);context_ids=encode('\n'.join(lines[offset:]+lines[:offset]).rstrip()+'\n')
result={'schema':'mtplx-server-cell-prompt-ids-v1','model':str(artifact),'model_family':'deepseek-v41','context_sha256':digest(context.encode()),'instruction':instruction,'template_settings':{'add_generation_prompt':True,'enable_thinking':False,'add_special_tokens':False,'bos_token_id':0},'tokenizer_sha256':digest((artifact/'tokenizer.json').read_bytes()),'template_sha256':digest((artifact/'chat_template.jinja').read_bytes()),'prompts':[]}
for target in (1024,16384):
 budget=target-len(encode(instruction['prompt']))-4
 def build(n):
  text=tokenizer.decode(context_ids[:n],skip_special_tokens=False).rstrip()+'\n\n'+instruction['prompt']
  msgs=[{'role':'user','content':text}]
  rendered=render_deepseek_v41_prompt(msgs,enable_thinking=False,add_generation_prompt=True,drop_thinking=False)
  assert rendered==template.render(messages=msgs,enable_thinking=False,add_generation_prompt=True,preserve_thinking=True)
  return text,rendered,encode(rendered)
 for _ in range(12):
  text,rendered,ids=build(budget)
  if len(ids)==target:break
  budget+=target-len(ids)
 else:raise RuntimeError('exact prompt sizing failed')
 assert len(ids)==target and ids[0]==0 and ids[-2:]==[tokenizer.token_to_id('<｜Assistant｜>'),tokenizer.token_to_id('</think>')],(len(ids),ids[-5:])
 assert 'No code.' not in text and 'Return only a unified diff' in text
 (output/f'python-{target}.txt').write_text(text)
 (output/f'python-{target}.rendered.txt').write_text(rendered)
 entry={'cell':'sweep','target_tokens':target,'seed':seed,'text_sha256':digest(text.encode()),'templated_tokens':len(ids),'input_tokens':len(ids),'token_ids_sha256':digest(json.dumps(ids).encode()),'bos_id_prepended':True,'token_ids':ids}
 result['prompts'].append(entry);print(target,entry['token_ids_sha256'])
(output/'python-prompt-ids.json').write_text(json.dumps(result,indent=2)+'\n')
