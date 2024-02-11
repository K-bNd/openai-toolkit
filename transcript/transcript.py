import requests
import os


IA_SERVER_URL = os.getenv("SERVER_URL", "")
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./media/")


def get_transcript(audio_file: str) -> dict:
    """Get transcript from audio file.

    :param audio_file: the audio file in url form (needs to be accesible through browser)
    :type audio_file: str
    """
    file = {"file": audio_file}
    response = requests.get(url=IA_SERVER_URL + "/transcript", json=file, timeout=30)
    return {"text": response.content.decode()}


def get_subtitles_into_file(audio_file: str, filename: str) -> None:
    """Write subtitles to file in VTT standard format.

    :param audio_file: the audio file in url form (needs to be accesible through browser)
    :type audio_file: str

    :param filename: The name of the file we write to
    :type filename: str
    """

    file = {"file": audio_file}
    response = requests.get(url=IA_SERVER_URL + "/subtitles", json=file, timeout=30)

    out_file = open(MEDIA_ROOT + filename, "w", encoding="utf-8")
    out_file.write(response.content.decode())
    out_file.close()
