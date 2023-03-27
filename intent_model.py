
import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer

# %%
df = pd.read_csv('intent_final.csv')
# df = pd.read_csv('final_df.csv')

df.head()

# %%
df['intent'].unique()

# %%
df.info()


# %%
# Initializing the BertTokenizer from the pre-trained 'bert-base-cased' model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# %%
# Initializing the arrays to store the tokenized text

X_input_ids = np.zeros((len(df), 32))
X_attn_masks = np.zeros((len(df), 32))

# %%
# A Function to generate the tokenized training data

def generate_tokens(df, ids, masks, tokenizer):
    # Looping through the 'prompt' column of the dataframe

    for i, text in tqdm(enumerate(df['prompt'])):
        # Tokenizing the text
        tokenized_text = tokenizer.encode_plus(
            text, # the text to encode.
            max_length=32, # add padding to all the sentences to make them of the same length
            # padding [PAD] tokens, are added to the end of the sequence. 
            # paddings are used to ensure that all sequences in a batch have the same length. 
            # The model treats padding as non-information and does not attend to it.
            # Without padding, input sequences with different lengths would have to be processed separately, 
            #  which would increase the complexity of the model and slow down the training process.
            truncation=True,  # Truncate the sequence to the max_length
            padding='max_length', 
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                        # CLS token is used to classify the entire sequence.
                        # SEP token is used to separate two different sentences.
                        # BERT uses the first token of every sequence for classification purposes.
                        # The first token of every sequence is always a special classification token.
                        # The classification token is not used for any other purpose.
            return_tensors='tf' # Return TensorFlow tensors
        )
        # Storing the tokenized text 
        ids[i, :] = tokenized_text.input_ids # return the input ids of the tokenized text 
        masks[i, :] = tokenized_text.attention_mask # return the attnenetion masks.
        #  the attention mask in BERT is used to control the attention mechanism
        #  and ensure that the model focuses on the relevant tokens in the input sequence.
    # Returning the tokenized text
    return ids, masks

# %%
# calling the function and Generating the tokens for all the data
X_input_ids, X_attn_masks = generate_tokens(df, X_input_ids, X_attn_masks, tokenizer)

# %%
tokenized_text = tokenizer.convert_ids_to_tokens(X_input_ids[60])

# Print the tokenized text
print(tokenized_text)

# %%
# Initializing the array to store the labels
# second argument is the number of lables(intents)
labels = np.zeros((len(df), 4))
labels.shape

# %%
# Converting the 'intent_id' column of the dataframe to one-hot encoding
# The labels are populated using the values from the 'intent_id' column of the dataframe.
# np.arange(len(df)) to create an array with the same length as the dataframe "df",
# df['intent_id'].values gets the values of the "intent_id" column.
# more about one-hot could be found in the resources 
labels[np.arange(len(df)), df['intent_id'].values] = 1 

# %%
# Converting the data into a tensorflow dataset

dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))
# Taking the first sample from the dataset

dataset.take(1) 

# %%
# A Function to map the data into a dictionary
#it maps the data to a dictionary that includes the input ids and attention masks, and the labels
def DatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels

# %%
# Map the data into a dictionary
dataset = dataset.map(DatasetMapFunction)

# %%
dataset.take(1)


# %%
# The dataset is shuffled using the 'shuffle' method (randomly rearranges the order of the elements)(with a buffer size of 10000)
dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)
# A batch: is a subset of the training data used in one iteration of the training process
# Instead of training the model on the entire training dataset, the model is trained on small groups of samples.
# for memory and computational efficiency, 
# and Stochastic Behavior: Training on small batches introduces some randomness into the training process, 
# which can help the model to have a better generalization and less local minimum .
# so here batch is used to divide the dataset into batches of 16 samples each 
# and drop_remainder=True is used to drop the last batch if it has fewer than 16 samples.
# This can cause in a loss of data, if the size of the dataset is small and the batch size is large.
# but also it may not have any impact on the loss
# because over the course of many iterations, the model will loop through all the samples in the training dataset.
# 

# %%
# The ratio of the size of the training set
# The training set is 70% of the dataset
ratio = 0.7
train_size = int((len(df)//16)*ratio)

# %%
# The dataset is divided into a training set and a validation set using the 'take' and 'skip' methods. 

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# %%
from transformers import TFBertModel

# %%
# Initializing the Bert model
model = TFBertModel.from_pretrained('bert-base-cased')

# %%
# Building the neural network
# Two input layers are created to store the input ids and attention masks
# The input ids and attention masks are passed to the BERT model to obtain the pooled output layer
# The pooled output layer is passed to a dense layer with 64 neurons and a ReLU activation function
# The output of the dense layer is passed to a dense layer with 11 neurons and a softmax activation function
# The softmax activation function is used to calculate the probabilities of the classes

# The relu activation function is used to add non-linearity into the model
# the softmax activation function is used to calculate the probabilities of the different classes.

# the input layers has a shape of (32,) which is a 1-dimensional tensor with the maximum length of 256 tokens
# 
input_ids = tf.keras.layers.Input(shape=(32,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(32,), name='attention_mask', dtype='int32')

bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1] 
# The [1] indexing is used to extract the second element from the output of the bert method,
# which is the "pooled output layer"
# The "pooled output layer" is a feature representation obtained from a pre-trained BERT model.
intermediate_layer = tf.keras.layers.Dense(64, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(4, activation='softmax', name='output_layer')(intermediate_layer) 
# the output layer has a shape of (4,) because there are 11 classes

# compiling the model with all the layers:
intents_model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
intents_model.summary()

# %%
# Defining the optimizer, loss function, and accuracy metric
# The Adam optimizer is used to minimize the loss function
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, decay=1e-5)
loss_func = tf.keras.losses.CategoricalCrossentropy()
accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')
f1= tf.keras.metrics.CategoricalAccuracy('f1')
crossEtropy = tf.keras.metrics.CategoricalCrossentropy('crossEtropy')


# %%
# The model is compiled using the Adam optimizer and the sparse categorical crossentropy loss function

intents_model.compile(optimizer=adam_optimizer, loss=loss_func, metrics=[accuracy,f1,crossEtropy])

# %%
tf.keras.utils.plot_model(intents_model)


# %%
import matplotlib.pyplot as plt

# # training_history: Training the model
# The model is trained for 15  epochs
# The training dataset is passed to the model using the 'fit' method
# The validation dataset is passed to the model using the 'validation_data' parameter
# The model is evaluated on the validation dataset after each epoch

training_history = intents_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15 
)
#plotting the training history



# %%
# summarize history for accuracy
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
# summarize history for loss
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
results = intents_model.evaluate(val_dataset)

# %%
# Save the model
#intents_model.save('intent_model')

# %%
# intents_model = tf.keras.models.load_model('model')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# preparing the data that will be passed to the model as an input for testing
def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=32, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['performance', 'connection', 'access', 'sonstiges']):
    # The predict method is used to pass the processed_data to the neural network
    #  and obtain the predicted class probabilities.
    # The result of the predict method is a 2-dimensional array
    # the first dimension represents the number of samples in processed_data 
    # the second dimension represents the number of classes in the output layer.
    # The indexing [0] is used to extract the predicted class probabilities
    # The np.argmax method is used to extract the index of the class with the highest probability
    # The classes map the index to its corresponding class name
    probs = model.predict(processed_data)[0]
    # returns the class with the highest probability
    return classes[np.argmax(probs)]
     

# %%
# input_text = input('Enter prompt here: ')
# processed_data = prepare_data(input_text, tokenizer)
# result = make_prediction(intents_model, processed_data=processed_data)
# print(f"Predicted intent: {result}")

# %%
# input_list=['still my printer is not displaying to an internal error message that simply says printer is not connectedeven though though today it is connected to all my computer.	','i cant connect to my email', 'I got no internet',  ' I dont have internet', ]
input_list = ["my monitor says theres no signal", "my printer gives empty cartridge error"]
def predict_list (input_list):
    for i in input_list:
        processed_data = prepare_data(i, tokenizer)
        result = make_prediction(intents_model, processed_data=processed_data)
        print(f"prompt: {i}, Predicted intent: {result}")

predict_list(input_list)

# %%
def answerMsgBert(msg):
    processed_data = prepare_data(msg, tokenizer)
    result = make_prediction(intents_model, processed_data=processed_data)
    return result


# %%
answerMsgBert("my monitor says theres no signal")


