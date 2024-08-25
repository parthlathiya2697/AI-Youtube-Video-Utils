# Fastapi Application
# Description: This file contains the Fastapi application that is used to serve the summarisation model to youtube video links
HOST = '127.0.0.1'
PORT = 5000


# Imports
import uvicorn
from fastapi import FastAPI
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from youtube_summariser_backend.youtube_summariser_utils import YoutubeSummariserUtils, SummarisationTechniques

# Schemas
from pydantic import BaseModel
class VideoTextSchema(BaseModel):
	url: str

	summarisation_technique : Optional[str] = 'summarise_text_openai'

	class Config:
		schema_extra = {
			"examples": {
				"url": "https://www.youtube.com/watch?v=6QapdLd39A0"
			}
		}


# Create and start app in main
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Home api
@app.get('/')
def home():
    return {'message': 'Welcome to the Youtube Summariser API'}


@app.post('/fetch_video_captions')
def fetch_video_text(request: VideoTextSchema):
	if not request.url:
		raise ValueError('Please provide a valid youtube video url')
	subtitles_text = None
	youtubeSummariserUtils = YoutubeSummariserUtils(url= request.url)
    
	try:
		captions = youtubeSummariserUtils.get_video_captions()
		if not captions:
			raise ValueError('Could not fetch video captions')
		subtitles_text = " ".join( [ caption['text'] for caption in captions ] )
	except:
		
		print(f':::::::: Transcripts disabled for {request.url} ::::::::::')
		print(f'Trying to get captions using ML')

		subtitles_text = youtubeSummariserUtils.get_youtube_captions_using_whisper()


	return {'data': subtitles_text }

@app.post('/fetch_video_summary')
def fetch_video_summary(request: VideoTextSchema):
	if not request.url:
		raise ValueError('Please provide a valid youtube video url')

	video_text = fetch_video_text(request).get('data', None)
	if not video_text:
		raise ValueError('Could not fetch video text')
	
	youtubeSummariserUtils = YoutubeSummariserUtils(url= request.url)
	print(f'request.summarisation_technique : {request.summarisation_technique}, ')
	if request.summarisation_technique == SummarisationTechniques.summarise_text_tfidf:
		summary = youtubeSummariserUtils.summarise_text_tfidf(video_text)
	elif request.summarisation_technique == SummarisationTechniques.summarise_text_bart:
		summary = youtubeSummariserUtils.summarise_text_bart(video_text)
	elif request.summarisation_technique == SummarisationTechniques.summarise_with_openai:
		summary = youtubeSummariserUtils.summarise_text_openai(video_text)
	else:
		raise ValueError(f'Please provide a valid summarisation technique. Valid techniques are: {list(SummarisationTechniques)}')

	return {'data': summary }


# Run app
if __name__ == '__main__':
	uvicorn.run('youtube_summariser_server:app', host= HOST, port= PORT, reload=True)