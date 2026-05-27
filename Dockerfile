# Start from RunPod's official optimized PyTorch CUDA environment
FROM runpod/base:0.4.0-cuda11.8.0-py310

# Set the working environment directory inside the cloud container
WORKDIR /

# Install system audio utilities
RUN apt-get update && apt-get install -y git libsndfile1 && apt-get clean

# Copy dependencies list and utilize fast installing methods
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create internal structure folders and populate the default model cache assets
RUN mkdir -p model voice core
RUN wget -O model/kokoro.onnx https://huggingface.co/PatnaikAshish/kokoclone/resolve/main/model/kokoro.onnx
RUN wget -O voice/voices-v1.0.bin https://huggingface.co/PatnaikAshish/kokoclone/resolve/main/voice/voices-v1.0.bin
RUN wget -O core/chunked_convert.py https://raw.githubusercontent.com/Ashish-Patnaik/kokoclone/main/core/chunked_convert.py

# Copy your local script handler inside
COPY handler.py .

# Run the serverless engine loop on startup
CMD [ "python", "-u", "/handler.py" ]