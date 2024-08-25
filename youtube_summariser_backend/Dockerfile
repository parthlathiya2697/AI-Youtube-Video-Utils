# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /youtube_summariser_backend

# Copy the requirements file
COPY youtube_summariser_requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r youtube_summariser_requirements.txt

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the project files to the working directory
COPY . .


# Expose the port that the application runs on
EXPOSE 80