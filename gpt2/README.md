# gpt2-finetuned-nl2html

This code repository illustrates the fine-tuning of **gpt2** to generate HTML code using **vanilla PyTorch** code. <br>
**DISCLAIMER**: Finetuned version of the model is does not generate HTML code of the highest quality though it understands the structure of the code. 
This is because of the following factors:
* Minimal training data
* Lack of customized tokenizer for tokenizing the code data
* GPT2 being a small model

With that out of the way, let me walk through the process

 # Fine-Tuning
The script for fine-tuning is available in the **train.py** script. <br>
Steps followed in the training script:
* Step 1: Augment the data to the following format: ```Question: <NL-PROMPT> Answer: <HTML-CODE>```
* Step 2: Tokenize and concatenate the data
* Step 3: Decide on the BLOCK_SIZE, BATCH_SIZE, TRAIN_ITERS, EVAL_ITERS depending on the compute power at hand
* Step 4: Train and save the model

* Hyperparameters:
  - ```learning_rate=1e-4``` - setting it too high would damage the pre-trained weights
  - ```weight_decay=1e-3``` -  weight regularisation
  - ```TRAIN_ITERS=200``` - training for more number of epochs would overfit the model as we have only 528 training instances
  - ```EVAL_ITERS=20```
# Evaluation Results
* The evaluation results are shown below <br>
<img width="255" alt="Screenshot 2023-12-23 at 9 47 36 PM" src="https://github.com/SkAndMl/nl2html/assets/86184014/c0222a55-75a4-4ed3-b414-9619c00ceb7f">
<br>
* As we can see from the results, the loss keeps going down and we can even train it for more steps.
