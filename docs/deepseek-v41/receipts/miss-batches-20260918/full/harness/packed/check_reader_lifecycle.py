"""Two bounded CPU checks for early-read ownership after the measured probe win."""
import ast
from concurrent.futures import Future,ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS

root=Path(__file__).resolve().parent
source=ast.parse((root/'plane_lane.py').read_text())
names={'PlanePart','PartExecutor','ReaderExecutor','bind_reader'}
selected=[n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
ns=dict(Future=Future,threading=threading)
exec(compile(ast.Module(body=selected,type_ignores=[]),'plane-read-ownership','exec'),ns)
results=[]
for fail in (False,True):
    gate=threading.Event()
    local=threading.local()
    views=[]
    terminal=[]
    record=NS(expert=3,sidecar_offset=0)
    class Destination:
        def component_view(self,name):
            view=memoryview(bytearray(8))
            views.append(view)
            return view
    def read_range(path,offset,bufs,**kwargs):
        if offset==12533760:
            assert gate.wait(2),'down release not signaled'
            terminal.append('down')
            if fail:raise OSError('injected down failure')
        else:terminal.append(offset)
        bufs[0][0]=7
    with ThreadPoolExecutor(1) as splitting,ThreadPoolExecutor(2) as reading,ThreadPoolExecutor(2) as fanout:
        reader=NS(_readv_range_into=read_range,_fanout_executor=fanout,metrics=NS(update=lambda **k:None))
        ns['bind_reader'](reader,local)
        io=ns['ReaderExecutor'](reading,local)
        split=ns['PartExecutor'](splitting,local)
        destination=Destination()
        def ensure(layer,plan):
            f=io.submit(reader.read_record_into,None,record,destination)
            f.result()
            return NS(bindings=(NS(expert=3,buffer=destination),))
        full=split.submit(ensure,20,NS(loads=(NS(expert=3),)))
        part=split.parts[0][1]
        assert dict(part.gate_up_ready.result(timeout=2))=={3:destination}
        assert not full.done(),'record published before down terminal'
        assert all(len(view)==8 for view in views),'writer view released before down terminal'
        gate.set()
        try:
            full.result(timeout=2)
        except OSError as exc:
            assert fail and str(exc)=='injected down failure'
        else:assert not fail
        assert 'down' in terminal
        for view in views:
            try:len(view)
            except ValueError:pass
            else:raise AssertionError('writer view still owned after terminal read')
    results.append(dict(injected_down_failure=fail,early_gu_observed=True,full_record_waited_for_down=True,all_views_released=True))
report=dict(mlx_imported=False,cases=results)
(root/'reader-lifecycle-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
