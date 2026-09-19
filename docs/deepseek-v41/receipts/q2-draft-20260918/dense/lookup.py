"""Causal extensions of a complete native five-token proposal."""
from collections import defaultdict


class LookupExtension:
    def __init__(self, prompt, *, minimum_context=2, extra_tokens=2):
        self.history = list(prompt)
        self.ends = defaultdict(list)
        self.indexed_end = 5
        self.minimum_context = minimum_context
        self.extra_tokens = extra_tokens

    def append_committed(self, tokens):
        self.history.extend(tokens)
        for end in range(self.indexed_end, len(self.history)):
            self.ends[tuple(self.history[end-5:end])].append(end)
        self.indexed_end = len(self.history)

    def extend(self, native_draft):
        proposal = tuple(native_draft)
        best_context = self.minimum_context - 1
        best_end = None
        history = self.history
        # Occurrences are chronological. Equal context keeps the earliest one,
        # fixed by the first-half opportunity screen rather than future output.
        for end in self.ends.get(proposal, ()):
            context = 0
            while (context < 32 and end-5-context > 0
                   and history[end-6-context] == history[-1-context]):
                context += 1
            if context > best_context:
                best_context, best_end = context, end
        if best_end is None:
            return list(native_draft)
        return list(native_draft) + history[best_end:min(best_end+self.extra_tokens,len(history))]
