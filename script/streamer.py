from transformers import TextStreamer
from queue import Queue

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.generated_tokens = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end)
        if stream_end:
            self.queue.put(self.generated_tokens)

    def put(self, value):
        flattened_tokens = value.flatten().tolist()
        self.generated_tokens.extend(flattened_tokens)
        return super().put(value)
