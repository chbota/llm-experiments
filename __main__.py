from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from accelerate import infer_auto_device_map, init_empty_weights

import sys
import torch


def auto_generator(modelConfig):
  device_map=None
  def initialize_model(model):
    config = AutoConfig.from_pretrained(MODEL['model'], trust_remote_code=True, revision=MODEL['revision'])

    print('loaded config for model {}'.format(MODEL['model']))
    tokenizer = AutoTokenizer.from_pretrained(MODEL['model'], trust_remote_code=True, revision=MODEL['revision'])
    print('created tokenizer for model {}'.format(MODEL['model']))

    with init_empty_weights():
      model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
      print('initialized model')
      model.tie_weights()
      device_map = infer_auto_device_map(model)
      print('device map', device_map)
      model = AutoModelForCausalLM.from_pretrained(MODEL['model'], offload_folder="offload", trust_remote_code=True, device_map=device_map, revision=MODEL['revision']).half()
      print('loaded model')

      return (model, tokenizer)

  def auto_impl():
    (model, tokenizer) = initialize_model(modelConfig)

    def generate(prompt):
      x = tokenizer(prompt, return_tensors='pt')
      x = x.to('cuda')
      print('encoded prompt')
      y = model.generate(**x, max_length=255, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
      response = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

      return response
    return generate
  
  return auto_impl


def pipeline_generator(model):
  def pipeline_impl():
    print('creating opt pipeline')
    generator = pipeline('text-generation', model=model, device="cuda:0")
    print('ready for prompt')
    def generate(prompt):
      return generator(prompt, max_length=100, batch_size=8, do_sample=True, top_p=0.95, top_k=4, temperature=0.4, num_return_sequences=1)[0]['generated_text']
    
    return generate
  return pipeline_impl

REPLIT = {
  'model': 'replit/replit-code-v1-3b',
  'revision': 'main',
  'make_generator': auto_generator('replit/replit-code-v1-3b')
}
OPT13 = {
  'model': 'facebook/opt-13b',
  'revision': 'main',
  'make_generator': pipeline_generator('facebook/opt-13b')
}
OPT6_7 = {
  'model': 'facebook/opt-6.7b',
  'revision': 'main',
  'make_generator': pipeline_generator('facebook/opt-6.7b')
}
OPT1_3 = {
  'model': 'facebook/opt-1.3b',
  'revision': 'main',
  'make_generator': pipeline_generator('facebook/opt-1.3b')
}
MPT7 = {
  'model': 'mosaicml/mpt-7b',
  'revision': 'main',
  'make_generator': auto_generator('mosaicml/mpt-7b')
}
STARCODER = {
  'model': 'bigcode/starcoder',
  'revision': 'main',
  'make_generator': auto_generator('bigcode/starcoder')
}
STABLELM = {
  'model': 'StabilityAI/stablelm-tuned-alpha-3b',
  'revision': 'main',
  'make_generator': auto_generator('StabilityAI/stablelm-tuned-alpha-3b')
}


MODEL = REPLIT

generate = MODEL['make_generator']()


print(">>> ", end="", flush=True)
for line in sys.stdin:
  for word in generate(line):
    print(word, end="", flush=True)
  print()
  print(">>> ", end="", flush=True)


