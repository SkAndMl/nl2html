# nl2html

This code repository illustrates 
* Fine-tuning **LLaMa-2-7b** to generate HTML code corresponding to the natural language input given
* Deploying the model as a **API** using FastAPI

 This README file walks one through the complete pipeline for setting up the API

 # Fine-Tuning
The script for fine-tuning is available in the **train.py** script. <br>
The script uses the following techniques to train a 7B model in a single GPU
* BitsAndBytes -> To load the model weights and activations in 4 bits and for the inference time 16 bits are used. This quantization configuration makes sure that model can be trained on a T4 GPU with 16GB RAM
* LORA -> LORA is used for reducing the large number of computations involved in the matrix multiplications of the LLM by introducing new matrices that represent the weight updations of the model's weights which are of much lesser dimensions than the original weight matrices.
* Formatting the data -> To train LLaMa-2-7b model the authors used the following format ```<s>[INST] instruction-text [/INST] expected-output </s>```. To fine-tune the model we augment the dataset to follow the same format as it gives better results than just simply feeding in the data.
* Hyperparameters:
  - ```learning_rate=2e-4``` - setting it too high would damage the pre-trained weights
  - ```weight_decay=1e-3``` -  weight regularisation
  - ```epochs=1``` - training for more number of epochs would overfit the model as we have only 528 training instances

# Evaluation Results
* The number of training instances is 528 and splitting it further into a validatipn/test split would not be ideal for fine-tuning the model.
* Moreover, the LLaMa-2-7b model is capable of already generating decent HTML code, so finetuning it with more data is only going to enhance the HTML code generating capabilities of the model
* The training results are shown below
<img width="515" alt="Screenshot 2023-12-23 at 11 10 05 AM" src="https://github.com/SkAndMl/nl2html/assets/86184014/c4179975-81c2-4cb8-b54a-5776f0754c52"> <br>
* As we can see from the results, the loss keeps going down and we can even train it for more steps.

# API
The API deployment has been done using **FastAPI** and **ngrok**. <br>
* ngrok has been used to get a public url
* FastAPI has been used because it's suitable for developing a API quickly

## Steps to run the API
To get the API up and running follow the following steps
* Run the ```api.py``` script from the terminal and copy the public url that it prints out
* Paste the public url returned by the ```api.py``` file and paste it in the ```<PUBLIC-URL>``` part of the ```query.py``` file
* Run the query.py file and enter any queries related to HTML
Here's a sample output
<img width="1095" alt="Screenshot 2023-12-23 at 11 06 33 AM" src="https://github.com/SkAndMl/nl2html/assets/86184014/fa634df5-849f-44d8-a4b9-ae440b4bc8e3">
