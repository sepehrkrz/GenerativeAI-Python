import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=0  # Specify GPU device if available
    )
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]
    return result

def summerize_keypoints(transcript):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"List the key points with details from the context: Context: {transcript}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids)
    keypoints = tokenizer.decode(outputs[0])
    return keypoints

def keypoint_audio(audio_file):
    audio_trans = transcript_audio(audio_file)
    output = summerize_keypoints(audio_trans)
    return output

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=keypoint_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)
