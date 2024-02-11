from copy import deepcopy
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from openai import OpenAI
import requests
import tiktoken

from langchain.text_splitter import TokenTextSplitter
from transcript.transcript import get_transcript


class Summary:
    """
    Class to generate summaries with the OpenAI API.
    """

    GPT3_MAX_TOKEN = 4097
    GPT3_16_MAX_TOKEN = 16385
    TOKENS_PER_MESSAGE = 3
    BASE_NUM_TOKENS = 3

    def __init__(self, api_key) -> None:
        """
        Init function

        :param api_key: The API key to access the OpenAI API.
        :type api_key: str
        """

        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.prev_text = ""
        self.auto_prompt = """
            You are a helpful assistant who is given content
            from various sources (audio, text or urls).
            Your task is to summarize using
            both extractive and abstractive summarization as follows:
            - Overall summary of the content
            - If applicable, a list of subjects that need more explanations.
            """
        self.extract_summary_prompt = """
            You are a helpful assistant who is given content
            from various sources (audio, text or urls).
            Your task is to summarize using extractive summarization as follows:
            - Overall summary of the content
            - If applicable, a list of subjects that need more explanations.
            """
        self.extract_prompt_length = len(
            self.encoding.encode(self.extract_summary_prompt)
        )

    def generate_recursive_summary(self, text: str, debug=False) -> str:
        """
        Generate a summary by breaking the text into chunks

        :param text: The input text to be summarized.
        :type text: str

        :param debug: Whether to print debug information. Default is False.
        :type debug: bool

        :return: The summarized text.
        :rtype: str
        """

        # The API takes into account the encoding of the entire message
        #         {"role": "system", "content": prompt},
        #         {"role": "user", "content": text},
        # the values ('system' and 'user') are also encoded (each count for one token)
        # We also encode the each message (tokens_per_message = 3)
        # Consult:
        # https://github.com/openai/openai-cookbook/blob/4d373651822c3a27290078d713f14eeb1d8f5d3d/examples/How_to_count_tokens_with_tiktoken.ipynb

        misc_token_length = 2 * Summary.TOKENS_PER_MESSAGE + Summary.BASE_NUM_TOKENS + 2
        chunk_size = (
            Summary.GPT3_16_MAX_TOKEN - self.extract_prompt_length - misc_token_length
        )

        token_splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=chunk_size - 1,
            chunk_overlap=0,
        )
        texts = token_splitter.split_text(text)
        text_summaries = []

        for chunk_text in texts:
            text_summaries.append(
                self.__call_gpt(
                    chunk_text, prompt=self.extract_summary_prompt, debug=debug
                )
            )

        summarized_text = "\n".join(text_summaries)
        return self.__call_gpt(summarized_text, debug=debug)

    def __pick_model(self, tok_length: int) -> str:
        """
        Find the right gpt model based on the amount of tokens.

        :param tok_length: The quantity of tokens that need to be processed.
        :type tok_length: int

        :return: The model required to process the content.
        :rtype: str
        """

        if tok_length < Summary.GPT3_MAX_TOKEN:
            return "gpt-3.5-turbo"
        return "gpt-3.5-turbo-16k"

    def __call_gpt(self, text: str, prompt=None, debug=False) -> str:
        """
        Call the API to get the summary

        :param text: The text to use as input for the model.
        :type text: str

        :param prompt: The prompt to use to guide the model. If not provided, the default prompt is used.
        :type prompt: str or None

        :param debug: A boolean flag to print the chosen model and the token quantity. Default is False.
        :type debug: bool

        :return: The GPT-3 API response containing the generated summary.
        :rtype: str
        """

        if prompt is None:
            prompt = self.auto_prompt

        # The length of the prompt is counted in the token limit.
        tok_length = len(self.encoding.encode(text + prompt))

        if tok_length >= Summary.GPT3_16_MAX_TOKEN:
            if debug:
                print(f"Splitting text of length {tok_length}")
            return self.generate_recursive_summary(text, debug=debug)

        self.prev_text = deepcopy(text)
        model = self.__pick_model(tok_length)

        if debug:
            print(tok_length, model)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )

        return response.choices[0].message.content

    def pdf_summary(self, filename: str, prompt=None, debug=False) -> str:
        """
        Generate a summary from a PDF using the OpenAI API.

        :param filename: The name of the file to summarize.
        :type filename: str

        :param prompt: The prompt to use to guide the model. If not provided, the default prompt is used.
        :type prompt: str or None

        :param debug: A boolean flag to print debugging information. Default is False.
        :type debug: bool

        :return: The summary generated by the GPT-3 API.
        :rtype: str
        """

        return self.__call_gpt(extract_text(filename), prompt, debug)

    def url_summary(self, url: str, prompt=None) -> str:
        """
        Generate a summary from a URL using the OpenAI API.

        :param url: The URL to summarize.
        :type url: str

        :param prompt: The prompt to use to guide the model. If not provided, the default prompt is used.
        :type prompt: str or None

        :return: The summary of the article generated by the GPT-3 API.
        :rtype: str
        """

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return """
            URL error. We have failed to access the content.
            """
        soup = BeautifulSoup(response.content, "html.parser")
        word_list = soup.get_text().split()
        text = " ".join(word_list)
        return self.__call_gpt(text, prompt)

    def audio_summary(self, filename: str, prompt=None) -> str:
        """
        Generate a summary from an audio file.

        :param filename: The name of the file to summarize.
        :type filename: str

        :param prompt: The prompt to use to guide the model. If not provided, the default prompt is used.
        :type prompt: str or None

        :return: The summary of the audio generated by the GPT-3 API.
        :rtype: str
        """

        transcript = get_transcript(filename)
        return self.__call_gpt(transcript["text"], prompt)

    def rerun_summary(self, prompt: str) -> str:
        """
        Use the previous text with a new prompt.

        :param prompt: The prompt to use to guide the model. If not provided, the default prompt is used.
        :type prompt: str or None

        :return: A new summary of the previous text.
        :rtype: str
        """
        if not self.prev_text:
            return "Last content field empty: No reruns can be done"

        return self.__call_gpt(self.prev_text, prompt)
