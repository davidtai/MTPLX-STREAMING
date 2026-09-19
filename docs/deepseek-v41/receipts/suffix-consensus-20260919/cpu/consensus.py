"""Causal consensus backoff after a complete native five-token proposal."""
from collections import defaultdict


class SuffixConsensus:
    def __init__(self, prompt, *, min_suffix=4, min_count=2, max_extra=2):
        self.history = list(prompt)
        self.ends = defaultdict(list)
        self.indexed_end = 0
        self.lengths = tuple(range(6, min_suffix-1, -1))
        self.min_count = min_count
        self.max_extra = max_extra

    def append_committed(self, tokens):
        self.history.extend(tokens)
        for end in range(self.indexed_end, len(self.history)):
            for length in self.lengths:
                if end >= length:
                    self.ends[tuple(self.history[end-length:end])].append(end)
        self.indexed_end = len(self.history)

    def extend(self, native):
        context = self.history[-6:] + list(native)
        for length in self.lengths:
            occurrences = self.ends.get(tuple(context[-length:]), ())
            if len(occurrences) < self.min_count:
                continue
            suffix = []
            live = list(occurrences)
            for step in range(self.max_extra):
                live = [end for end in live if end+step < len(self.history)]
                if len(live) < self.min_count:
                    break
                token = self.history[live[0]+step]
                if any(self.history[end+step] != token for end in live[1:]):
                    break
                suffix.append(token)
            if suffix:
                return list(native)+suffix
        return list(native)
