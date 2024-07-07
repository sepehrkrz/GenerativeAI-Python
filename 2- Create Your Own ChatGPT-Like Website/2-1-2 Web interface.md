
# Integrating your Chatbot into a Web Interface


## Introduction

In this lab, you will learn to set up a back-end server and integrate your chatbot into a web application.

## Learning objectives

After completing this lab, you will be able to:

- Set up your back-end server
- Integrate your chatbot into your Flask server
- Communicate with the back-end using a web page

## Prerequisites

This section assumes you know how to build the simple terminal chatbot explained in the first lab.

There are two things you must build to create your ChatGPT-like website:

1. A back-end server that hosts your chatbot
2. A front-end webpage that communicates with your back-end server

Without further ado, let's get started!

### Step 1: Hosting your chatbot on a backend server

#### What is a backend server?

A backend server is like the brain behind a website or application. In this case, the backend server will receive prompts from your website, feed them into your chatbot, and return the output of the chatbot back to the website, which will be read by the user.

#### Hosting a simple backend server using Flask

**Note:** Consider using a requirements.txt file.

Flask is a Python framework for building web applications with Python. It provides a set of tools and functionalities to handle incoming requests, process data, and generate responses, making it easy to power your website or application.

#### Prerequisites

For all terminal interactions in this lab (such as running python files or installing packages), you will use the built-in terminal that comes with Cloud IDE. You may launch the terminal by either:

- Pressing Ctrl + `,` , or
- By selecting Terminal –> New Terminal from the toolbar at the top of the IDE window on the right.

In your terminal, let's install the following requisites:

```python
python3.11 -m pip install flask
python3.11 -m pip install flask_cors
```
# Setting up the server

Next, you will create a script that stores your Flask server code.

1. Create a new file in the directory `/home/project` by pressing Ctrl + N or by clicking 'File -> New File' in the IDE. Name the file `app.py`.

Let's take a look at how to implement a simple Flask server:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
```
# Saving and Running Your Flask Server

Paste the above code in the `app.py` file you just created and save it.

### Explanation of the Code:

In this code:

- You import the `Flask` class from the `flask` module.
- You create an instance of the `Flask` class and assign it to the variable `app`.
- You define a route for the homepage by decorating the `home()` function with the `@app.route()` decorator. The function returns the string `'Hello, World!'`. This means that when the user visits the URL where the website is hosted, the backend server will receive the request and return `'Hello, World!'` to the user.
- The `if __name__ == '__main__':` condition ensures that the server is only run if the script is executed directly, not when imported as a module.
- Finally, you call `app.run()` to start the server.

### Running the Server:

To run the server, execute the following command in your terminal:

```bash
python3.11 app.py
```
# Running Your Flask Server

Save this code in a Python file, for example, `app.py`, and run it by typing `python app.py` in the terminal. By default, Flask hosts the server at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

With this command, the Flask server will start running. If you run this server on your local machine, then you can access it by visiting [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or [http://localhost:5000/](http://localhost:5000/) in your web browser.

However, you are currently running this lab in the Skills Network Cloud. Thus, you can access your server as follows:

1. Navigate to the Skills Network Toolbox from the toolbar on the left side of the IDE.
2. Click "Launch Application" in the adjacent vertical sidebar.
3. Enter `5000` as your Application Port.
4. Launch the application in a new browser tab.

By performing the above steps, you have visited the relative localhost URL of the cloud server.

**IMPORTANT:** Throughout the rest of this lab, you will refer to this URL as `<HOST>`.

On visiting the localhost, you should see the "Hello, World!" message displayed.

Here's what it should look like:

```plaintext
Hello, World!
```
# Adding Additional Routes

Let's add the following routes to try it out:

```python
@app.route('/bananas')
def bananas():
    return 'This page has bananas!'
    
@app.route('/bread')
def bread():
    return 'This page has bread!'
```
## Testing Additional Routes

1. Stop your app by pressing `Ctrl + C` in the terminal.
2. Re-run the app using `flask run`.
3. Visit both of these routes:
   - [http://<HOST>/bananas](http://<HOST>/bananas)
   - [http://<HOST>/bread](http://<HOST>/bread)

Here's what you should see:

---

Okay, now that you've demonstrated how routes work, you can remove these two routes (`bananas` and `bread`) from your `app.py` as you won't be using them.

Before proceeding, you'll also need to add two more lines of code to your program to mitigate CORS errors. CORS errors occur when making requests to domains other than the one hosting the webpage.

You'll be modifying your code as follows:
```python
from flask import Flask
from flask_cors import CORS		# newly added

app = Flask(__name__)
CORS(app)				# newly added

@app.route('/')
def home():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
```
# Integrating Your Chatbot into Your Flask Server

Now that you have your Flask server set up, let's integrate your chatbot into your Flask server.

As stated at the beginning, this lab assumes you've completed the first lab of this guided project on how to create your own simple chatbot.

### Installing Required Libraries

First, install the necessary libraries:

```bash
python3.11 -m pip install transformers torch
```
# Initializing Your Chatbot in Flask

Next, let’s copy the code to initialize your chatbot from lab 1 and place it at the top of your script. You also need to import the necessary libraries for your chatbot.

```python
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []
```
Next, you'll need to import a couple more modules to read the data.

```python
from flask import request
import json
```
Before implementing the actual function, you need to determine the structure you expect to receive in the incoming HTTP request.

Let's define your expected structure as follows:

```json
{
    'prompt': 'message'
}
```
# Implementing Your Chatbot Function

Now implement your chatbot function. Copy the code over from your chatbot implementation from the first lab.

```python
@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    
    # Create conversation history string
    history = "\n".join(conversation_history)
    
    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
    
    # Generate the response from the model
    outputs = model.generate(**inputs, max_length=60)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    
    return response
```
# Final Version of Your Flask Application

```python
from flask import Flask, request
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    
    # Create conversation history string
    history = "\n".join(conversation_history)
    
    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
    
    # Generate the response from the model
    outputs = model.generate(**inputs, max_length=60)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    
    return response

if __name__ == '__main__':
    app.run()
```
# Testing Your Flask Application with curl

To test your implementation, use `curl` to make a POST request to `<HOST>/chatbot` with the following request body: `{'prompt':'Hello, how are you today?'}`.

1. Open a new terminal:
   - Select terminal tab –> open new terminal

2. Execute the following `curl` command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}' 127.0.0.1:5000/chatbot
```
# Communicating with Your Backend Using a Webpage

In this section, you'll download a template chatbot webpage and configure it to make requests to your backend server.

First, let's clone a repository that has a template website and install your required libraries. If your Flask app is running, terminate it with Ctrl + C and run the following lines in the terminal:

```bash
git clone https://github.com/ibm-developer-skills-network/LLM_application_chatbot
python3.11 -m pip install -r LLM_application_chatbot/requirements.txt
```
`If the operations are complete with no errors, then you have successfully obtained a copy of the template repository.

The file structure of this repo should be as follows:` 

ibm-chatbot-template/ static/ script.js <other assets> templates/ index.html


Let's move your Flask app `app.py` to the `LLM_application_chatbot/` folder so that you can host `index.html` on your server.

Both `app.py` and the `LLM_application_chatbot/` folder should be in `/home/project`. You can move `app.py` into `LLM_application_chatbot/` by running the following line in the terminal:

```bash
mv app.py LLM_application_chatbot/
```
Now, let's modify your `app.py` so that you can host `index.html` at `<HOST>/`. You can achieve this by adding the following route to your code:

```python
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
```
After adding the code, you can run your Flask app with the following command:

```bash
flask run
```
This template already has JavaScript code that emulates a chatbot interface. When you type a message and hit send, the website template does the following:

- Enters your input message into a text bubble
- Sends your input to an endpoint (by default, this is set to www.example.com in the template)
- Waits for the response from the endpoint and puts the response in a text bubble

However, you will not implement such an interface as it is beside the purpose of this lab.

Instead, you will ensure that in step 2, the user input is sent to the route you created for your chatbot earlier: http://127.0.0.1:5000/chatbot.

To send the input, open `/static/script.js` and locate where the endpoint is set.

Let's change this endpoint to your chatbot route. Replace:

```javascript
'https://sinanz-5000.theianext-0-labs-prod-misc-tools-us-east-0.proxy.cognitiveclass.ai/chatbot'
```
The URL may be different for you. Basically, copy the URL from your app launch and add `/chatbot` at the end.

And that should be it! Before testing your code, let's glance at the final version of your Flask app:

```python
from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    print(data)  # DEBUG
    input_text = data['prompt']
    
    # Create conversation history string
    history = "\n".join(conversation_history)
    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
    # Generate the response from the model
    outputs = model.generate(**inputs)
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    
    return response

if __name__ == '__main__':
    app.run()
```
