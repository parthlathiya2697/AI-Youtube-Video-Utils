# Use a base image with the necessary dependencies for your application
FROM node:14.17.6

# Set the working directory
WORKDIR /youtube_summariser_frontend

# Copy the package.json and package-lock.json files
COPY package.json .
COPY package-lock.json .

# Install dependencies
RUN npm ci --only=production

# Copy the rest of the application code
COPY . .
