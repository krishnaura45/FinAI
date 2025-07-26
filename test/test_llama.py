# from llama_cpp import Llama

# llm = Llama(
#     model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
#     n_ctx=1024,
#     verbose=True
# )
# print("✅ Model loaded successfully!")

from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=256,
    n_ctx=1024,
    f16_kv=True
)

response = llm.predict("Who is the Prime Minister of India?")
print(response)