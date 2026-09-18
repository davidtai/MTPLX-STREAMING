"""Fixed-size coarse diagnostic spans; no MLX import or evaluation fences."""
import time


class Span:
    __slots__=('calls','wall_ns','thread_ns','process_ns','max_wall_ns')

    def __init__(self):
        self.calls=self.wall_ns=self.thread_ns=self.process_ns=self.max_wall_ns=0

    @staticmethod
    def start():
        return time.perf_counter_ns(),time.thread_time_ns(),time.process_time_ns()

    def finish(self,start):
        wall,thread,process=self.start()
        elapsed=wall-start[0]
        self.calls+=1
        self.wall_ns+=elapsed
        self.thread_ns+=thread-start[1]
        self.process_ns+=process-start[2]
        self.max_wall_ns=max(self.max_wall_ns,elapsed)
        return elapsed/1e9

    def snapshot(self):
        return {'calls':self.calls,'wall_s':self.wall_ns/1e9,
                'main_thread_cpu_s':self.thread_ns/1e9,'process_cpu_s':self.process_ns/1e9,
                'wall_minus_main_thread_cpu_s':(self.wall_ns-self.thread_ns)/1e9,
                'max_wall_s':self.max_wall_ns/1e9}


class Recorder:
    def __init__(self):
        self.phases={name:Span() for name in ('decode_cycles','draft','verify','accept','commit','target_forward','final_eval')}
        self.attention={layer:Span() for layer in range(40)}
        self.expert={layer:Span() for layer in range(40)}

    def start(self):
        return Span.start()

    def finish(self,name,start):
        return self.phases[name].finish(start)

    def snapshot(self):
        return {'scope':'Diagnostic elapsed and CPU clocks at existing coarse boundaries. Wall minus main-thread CPU includes worker activity, blocking and descheduling; it is not GPU-idle time. Entry-point wall times may charge deferred work from an earlier graph. Nested spans must not be added to parent totals.',
                'phases':{name:span.snapshot() for name,span in self.phases.items()},
                'attention_entrypoints':{str(layer):span.snapshot() for layer,span in self.attention.items()},
                'expert_entrypoints':{str(layer):span.snapshot() for layer,span in self.expert.items()}}
