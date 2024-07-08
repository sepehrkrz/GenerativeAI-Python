import torch
import os
import gradio as gr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate

# Initialize the speech recognition pipeline from Hugging Face Transformers
speech_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

# Initialize the text generation pipeline with Llama model
llama_model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the correct LLaMA model name
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

text_pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
)

#######------------- Prompt Template-------------####

temp = """
List the key points with details from the context: 
Context: {context}
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

# Define a custom LLMChain class to use the Hugging Face pipeline
class CustomLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, context):
        prompt_text = self.prompt.format(context=context)
        response = self.llm(prompt_text, max_length=150, num_return_sequences=1, temperature=0.7)
        return response[0]['generated_text'].strip()

prompt_to_model = CustomLLMChain(llm=text_pipe, prompt=pt)

#######------------- Speech2text-------------####

def transcript_audio(audio_file):
    # Transcribe the audio file and return the result
    transcript_txt = speech_pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_model.run(transcript_txt)
    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, 
    outputs=output_text, 
    title="Audio Transcription App",
    description="Upload the audio file"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
