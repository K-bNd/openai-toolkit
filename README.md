# OpenAI Tools for audio and text

- Set of tools for media summary and transcript \*

This codebase provides code to generate summaries and transcript for text and audio.
The submodule is a Docker container meant to run on a GPU for AI inference.

## Requirements

This project requires [Python 3.11](https://www.python.org/downloads/) to run as well as [Docker](https://www.docker.com/get-started/).

You will also need to the python packages in requirements.txt using [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

### Environment variables
You will need to have the following as environment variables:

- OPENAI_KEY: OpenAI API key for summaries
- SERVER_URL: URL where the API is accesible
- MEDIA_ROOT: Where files can be stored

The API submodule will also need to be deployed so check its README as well.

## Usage

```py
from summary import Summary
from transcript.transcript import get_transcript

openai_key = open("OPENAI_KEY.txt", "rb").readline().decode().rstrip()

summ = Summary(openai_key)

PDF_FILENAME = "./Lorem_ipsum.pdf"
AUDIO_FILENAME = "./test.mp4"

lorem_summary = summ.pdf_summary(PDF_FILENAME)
audio_summary = get_transcript(AUDIO_FILENAME)

print(f"PDF summary: {lorem_summary}\n MP4 summary: {audio_summary}\n")
```
