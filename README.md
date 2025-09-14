# RAG-LLM

## Project Description

This project implements a multimodal local chat application leveraging Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). It allows users to interact with the LLM through text input, voice recordings, uploaded audio and image files, and PDF documents. The application supports chat history management and PDF document querying using vector databases.

## Features and Functionality

*   **Multimodal Input:** Supports text, voice recordings (microphone input and audio file uploads), image uploads, and PDF document uploads.
*   **Chat History Management:** Saves and loads chat histories for persistent conversations.  Chat sessions are stored in the `chat_history_path` directory specified in `config.yaml`.
*   **PDF Chat:** Enables querying of uploaded PDF documents using a vector database.
*   **Image Description:**  Uses a local LLaVA model to describe images.
*   **Voice Transcription:** Transcribes audio input (microphone and uploaded files) using the Whisper model.
*   **RAG Implementation:**  Uses a Chroma vector database to store PDF embeddings and perform retrieval for PDF chat.
*   **Local LLM:** Employs a local LlamaCpp model for text generation.
*   **Streamlit UI:** Provides a user-friendly interface built with Streamlit.

## Technology Stack

*   **Python:** Primary programming language.
*   **Streamlit:** For creating the user interface.
*   **Langchain:** Framework for building LLM applications.
*   **LlamaCpp:**  For running the LLM locally.
*   **CTransformers:**  (Potentially - commented out code suggests it was considered) For running LLMs.
*   **Hugging Face Transformers:**  For audio transcription (Whisper) and embeddings.
*   **ChromaDB:** Vector database for storing PDF embeddings.
*   **pypdfium2:** PDF text extraction.
*   **librosa:** Audio processing library.
*   **streamlit-mic-recorder:** For microphone recording in Streamlit.
*   **YAML:** For configuration management.

## Prerequisites

Before running the application, ensure you have the following prerequisites:

*   **Python 3.7+**
*   **CUDA (Optional):** For GPU acceleration.
*   **Models:** Download the necessary models for image handling and text generation. Ensure the paths specified in `config.yaml` point to the correct locations.  Specifically, you'll need models for `Llava15ChatHandler` and `Llama`.

## Installation Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoozKelly/RAG-LLM.git
    cd RAG-LLM
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    If you encounter any issues with the `requirements.txt` file, you can manually install the required packages:

    ```bash
    pip install streamlit langchain llama-cpp-python transformers chromadb pypdfium2 librosa streamlit-mic-recorder pyyaml
    ```

4.  **Download Models:**
    Download the following models and place them in the locations specified in `config.yaml`:

    *   **LLaVA Model:**  `./models/llava/mmproj-model-f16.gguf` (for image handling)
    *   **LLM Model:** `./models/llava/ggml-model-q5_k.gguf` (for image handling and text generation)
    *   **Ensure the correct paths are set in `config.yaml`.**
    * It is also essential to configure the model path in `config.yaml` for LLM chains, using the "large" key. This model is used for the main chat and should also be downloaded and placed in the correct directory.

## Usage Guide

1.  **Configure the application:**

    *   Edit the `config.yaml` file to configure the following:
        *   `model_path`: Paths to your LLM models (both the general LLM and the LLaVA image model).  The `large` key specifies the primary LLM.
        *   `embeddings_path`: Path to the Hugging Face Instruct Embeddings model.
        *   `chat_history_path`:  Directory where chat history files are stored.
        *   `model_type`: (Potentially obsolete, check `llm_chains.py` for relevance). Model type for CTransformers (if used).
        *   `model_config`: Configuration parameters for the LLM, such as `gpu_layers`, `context_length`, `temperature`, and `max_new_tokens`.

2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

3.  **Interact with the application:**

    *   **Text Input:** Type your message in the text input field and press "Send".
    *   **Voice Recording:** Use the microphone recorder to record your voice.
    *   **Audio Upload:** Upload an audio file (wav, mp3, ogg).  The application will transcribe the audio and summarize it.
    *   **Image Upload:** Upload an image file (jpg, jpeg, png). The application will describe the image.
    *   **PDF Upload:** Upload one or more PDF files.  After processing, you can enable "PDF Chat" to query the documents.
    *   **Chat Sessions:** Select a chat session from the sidebar to load its history or start a new session.

## API Documentation

This project primarily uses Langchain's API for interacting with the LLM.  Refer to the Langchain documentation for details on the available methods and parameters. Key components used include:

*   `LLMChain`: For creating a chain of operations involving the LLM.
*   `RetrievalQA`: For question answering over documents using retrieval.
*   `HuggingFaceInstructEmbeddings`: For creating embeddings.
*   `Chroma`: For interacting with the Chroma vector database.

The `app.py` file provides examples of how to use these components within the Streamlit application.  See the `llm_chains.py` file for the creation and usage of these Langchain components.

## Contributing Guidelines

Contributions are welcome! To contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and descriptive commit messages.
4.  Submit a pull request.

## License Information

No license has been specified for this project. All rights are reserved by the author.

## Contact/Support Information

For questions or support, please contact [JoozKelly](https://github.com/JoozKelly) through GitHub.
