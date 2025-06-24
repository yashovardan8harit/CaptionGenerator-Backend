# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application's code into the container at /code
COPY . /code/

# Tell Docker that the container will listen on port 7860
EXPOSE 7860

# Define the command to run your app using uvicorn
# It listens on all interfaces (0.0.0.0) on the specified port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]