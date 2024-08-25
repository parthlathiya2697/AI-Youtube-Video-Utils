# Run Project using Docker
docker-compose build
docker-compose up


<!-- Set local Virtual Environment -->
virtualenv --python=python3.9 .venv
source .venv/bin/activate
pip install  -r youtube_summariser_requirements.txt

<!-- Start Backend Server -->
python3 youtube_summariser_backend/youtube_summariser_server.py


<!-- Start Frontend Server -->
cd youtube_summariser_frontend
npm start


<!-- Inputs to try -->
with subtitles : 
- https://www.youtube.com/watch?v=ry9SYnV3svc

without subtitles :
- https://www.youtube.com/watch?v=6QapdLd39A0