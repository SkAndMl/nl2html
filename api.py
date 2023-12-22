from flask import Flask, request
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import pipeline
import torch

BASE_MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
REFINED_MODEL = "llama-2-7b-nl2html" 
llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1


peft_model = PeftModel.from_pretrained(base_model, REFINED_MODEL)
text_gen = pipeline(task="text-generation", model=peft_model, tokenizer=llama_tokenizer, max_length=200)


app = Flask(__name__)

@app.route("/query", methods=["GET"])
def query_model():
    query = request.args.get("query")
    output = text_gen(f"<s>[INST] {query} [/INST]")
    return {"response": output[0]['generated_text']}

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)