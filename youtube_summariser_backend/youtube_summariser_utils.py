# Fixxed: SSL: CERTIFICATE_VERIFY_FAILED
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Default
import os
import logging
logging.basicConfig(filename='demo.log', level=logging.ERROR)

import os, json, requests, random


from enum import Enum

# Custom
import whisper
from pytube import YouTube

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from transformers import pipeline
import nltk
# nltk.download('punkt')
downloader = nltk.downloader.Downloader()
downloader.download('punkt')

from typing import Callable, List, Dict


# Gobal Variables
TEMP_DIR_AUDIO_FILES = './temp_audio_files'

API_KEYS = [] # Put your api keys here
API_KEY_SELF=''  # Put your api key here
API_KEYS_EXPIRED = []

class SummarisationTechniques(str, Enum):
	summarise_text_tfidf = 'summarise_text_tfidf'
	summarise_text_bart = 'summarise_text_bart'
	summarise_with_openai = 'summarise_text_openai'

class YoutubeSummariserUtils:

	# Indexing matters here
	WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

	def __init__(self, url: str = None, whisper_model_name: str = None):

		f'''
		Default: 
		- whisper_model_name: tiny
			Available Models : {YoutubeSummariserUtils.WHISPER_MODELS}
		'''

		# Validate
		if not url:
			raise ValueError('Please provide a valid youtube video url')

		self.url = url
		self.youtube_video_id = self.get_youtube_video_id()
		self.whisper_model_name = whisper_model_name if whisper_model_name else YoutubeSummariserUtils.WHISPER_MODELS[0]


	def get_youtube_video_id(self):
		return self.url.split("=")[-1]


	def download_audio_from_youtube(self, output_path: str = './temp_audio_files'):

		# Create output path if it doesn't exist
		os.makedirs(output_path, exist_ok=True)

		# Get Audio Stream
		video_url= YouTube(self.url)
		video = video_url.streams.filter(only_audio=True).first()

		# Download Audio to output folder (temporarily)
		output_filepath = os.path.join(output_path, self.get_youtube_video_id() + ".mp3")
		video.download(filename= output_filepath)
		
		return output_filepath


	def get_video_captions(self):
		try:
			return YouTubeTranscriptApi.get_transcript(self.youtube_video_id)
		except Exception as err:
			print(f'Exception in get_video_captions: {err}')
			return 


	def load_whisper_model(self):
		return whisper.load_model(self.whisper_model_name)


	def transcribe_audio_to_text(self, model, audio_path: str = None, language: str = "English"):

		if not model:
			raise ValueError('Please provide a valid whisper model')
		if not audio_path or not os.path.exists(audio_path):
			raise ValueError('Please provide a valid audio path')


		# Generate Audio
		text = model.transcribe(audio_path, fp16=False, language=language)

		# Delete Audio
		os.remove(audio_path)
		return text


	def get_youtube_captions_using_whisper(self):

		model = self.load_whisper_model()
		audio_path = self.download_audio_from_youtube(output_path= TEMP_DIR_AUDIO_FILES)
		result = self.transcribe_audio_to_text(model, audio_path)
		del model
		return result


	def split_text_into_chunks(self, document: str, max_tokens: int):
		if not document:
			return []

		chunks, current_chunk, current_length = [], [], 0

		try:
			sentences = nltk.sent_tokenize(document)
			if len(sentences) == 1:
				print(f'Length 1, Joining using words. Length before chunking : {len(sentences[0])}, Length after : {len(sentences[0][:max_tokens])}')
				chunks.append(' '.join( nltk.word_tokenize(sentences[0][:max_tokens])  ) )
			else:
				for sentence in sentences:
					print(f'\n\n\n---> Sentence : {sentence}')
					sentence_length = len(sentence)

					if current_length + sentence_length < max_tokens:
						current_chunk.append(sentence)
						current_length += sentence_length
					else:
						chunks.append(" ".join(current_chunk))
						current_chunk, current_length = [sentence], sentence_length

				if current_chunk:
					print(f'Text Chunk : {current_chunk[:10]}')
					chunks.append(" ".join(current_chunk))

			return chunks
		except Exception as e:
			logging.error(f"Error splitting text into chunks: {e}")
			return []


	def get_summary_bart(
		self, list_chunks: List[str], summarizer: Callable, summarization_params: Dict[str, int]
	):
		# Generate summaries for each text chunk
		try:
			summaries = [
				summarizer(chunk, **summarization_params)[0]["summary_text"]
				for chunk in list_chunks
			]
			return " ".join(summaries)
		except Exception as e:
			logging.error(f"Error generating summaries: {e}")
			return ""


	def summarise_text_bart(self, text):
		
		text_chunks = self.split_text_into_chunks(text, max_tokens=4000)
		bart_params = {
			"max_length": 124,
			"min_length": 30,
			"do_sample": False, 
			"truncation": True,
			"repetition_penalty": 2.0,
		}

		summarizer = pipeline("summarization", "google/pegasus-xsum") # facebook/bart-large-cnn
		tokenizer_kwargs = {'truncation':True,'max_length':512}
		short_summary = self.get_summary_bart(text_chunks, summarizer, bart_params)
		return short_summary
	

	def summarise_text_tfidf(self, text):

		'''
		##TODO: not working
		'''

		sentences = sent_tokenize(text)  # not proper
		organized_sent = {k:v for v,k in enumerate(sentences)}
		tf_idf = TfidfVectorizer(min_df=2, 
                                    strip_accents='unicode',
                                    max_features=None,
                                    lowercase = True,
                                    token_pattern=r'w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
		print(type(sentences), sentences)
		sentence_vectors = tf_idf.fit_transform(sentences)
		sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
		N = 3
		top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:N]]
		# mapping the scored sentences with their indexes as in the subtitle
		mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
		# Ordering the top-n sentences in their original order
		mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
		ordered_sentences = [element[0] for element in mapped_sentences]
		# joining the ordered sentence
		summary = " ".join(ordered_sentences)
		return sum


	def summarise_text_openai(self, text):

		print(f'Starting OpenAI Summarisation, length of text : {len(text)}')
	

		url = "https://api.openai.com/v1/chat/completions"

		text_summary = ''
		text_chunks = self.split_text_into_chunks(text, max_tokens=4000)
		print(f'Number of chunks : {len(text_chunks)}, Each chunk size : {[len(text_chunk) for text_chunk in text_chunks]}')
		for text in text_chunks:
			payload = json.dumps({
			"model": "gpt-3.5-turbo",
			"messages": [ {'role': 'user', 'content' : f'Summarise the following Text in the most accurate way possible :\nText: {text}\n\nSummary:'} ],
			"stream": False
			})
			

			global API_KEYS, API_KEYS_EXPIRED
			response_data = None
			retry_count = 0
			while len(API_KEYS):
				api_key = random.choice(API_KEYS)

				headers = {
					'Content-Type': 'application/json',
					'Authorization': f'Bearer {api_key}'
					}

				response = requests.request("POST", url, headers=headers, data=payload)
				response_data = response.json()
				if 'error' in response_data:
					print(f'[ FAILED from API KEYS] Try Count : {retry_count} with API Key : {api_key} Failed! \nException: {response_data}')

					# Removing API key
					API_KEYS_EXPIRED.append(api_key)
					API_KEYS.remove(api_key)
					retry_count += 1
				else:
					print(f'[ PASS from API KEYS] API Key : {api_key}.')
					text_summary_response =  response_data["choices"][0]["message"]['content']
					text_summary += text_summary_response
					break

			
			if 'error' not in response_data and not len(text_summary):
				api_key = API_KEY_SELF
				response = requests.request("POST", url, headers=headers, data=payload)
				response_data = response.json()
				if 'error' in response_data:
					print(f'[FAILED from API KEYS] Try Count : {retry_count} with API Key : {api_key} Failed! \nException: {response_data}')
				else:
					print(f'[ PASS from SELF API KEY] API Key : {api_key}.')
					text_summary_response =  response_data["choices"][0]["message"]['content']
					text_summary += text_summary_response

		
		
		return text_summary