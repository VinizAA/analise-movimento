# chat.py

import sys
import os
from llama_cpp import Llama

class SilenciarStderr:
    def __enter__(self):
        self.stderr_fileno = sys.stderr.fileno()
        self.null_fileno = os.open(os.devnull, os.O_WRONLY)
        self.saved_stderr = os.dup(self.stderr_fileno)
        os.dup2(self.null_fileno, self.stderr_fileno)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stderr, self.stderr_fileno)
        os.close(self.null_fileno)
        os.close(self.saved_stderr)

MODEL_PATH = "/home/vinicius/Downloads/llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_batch=512,
    use_mmap=True,
    use_mlock=False,
    chat_format="llama-2"
)

def chatbot_brain(messages):
    with SilenciarStderr():
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            stop=["</s>"]
        )
        
    return response["choices"][0]["message"]["content"]
