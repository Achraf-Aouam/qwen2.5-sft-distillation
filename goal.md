the goal of the enhanced eval watcher is to monitor the model with a new metric:
accuracy.

while the training code uses loss on 3 different datsets i want to add on those same datasets the accuracy of the model so after each model saved under outputs:

like outputs/checkpoint-20

the watcher will see the save and merge the model then run vllm on that model

then the work of the call funcyion is kicks in: we use the same: data/eval_order.json ....

to do an llm call to the model

for example with this prompt:

instruction prompt: "You are a data extraction assistant specialized in analyzing..... output valid JSON
user prompt: "Analyze the following document con....

these will be found on the json file on keys "instruction" and "input"

we will also load the "output" key as it has the GT we want and set it asside

after the LLM does its generation which in an ideal world would be valid json but might be not valid so this is where the utils_example has some ideas on how we can use code to "excuse" some of the unvalid json

like normalize keys for example: "modéle vehicule" = "modèle_vehicule" = "modelvehicule" etc....

and when the brackets don't fully close etc...

anyways after we fix json we will have 2 jsons at hand we look at GT key 1 and look at said key 1 in the llm output: if they are same good, if they are not same then wrong

specialy cases: if GT is null and json of llm has on that key and empty string or none or the key is ommited we consider these behaviors correct

and with this we calculate accuracy for the 3 datasets and we log them to wandb with the number of step

acc_order vs steps

acc_vehicle vs steps

acc_invoice vs steps

the accuracy is obviously a percentage
