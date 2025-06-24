# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Create a non-root user to run the application for better security
RUN useradd --create-home --shell /bin/bash appuser

# Create the persistent data directory and set its ownership
# This ensures the application has write permissions.
RUN mkdir -p /data && chown -R appuser:appuser /data

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your backend application code
COPY . /code/

# Change ownership of the code directory to the app user
RUN chown -R appuser:appuser /code

# Switch to the non-root user
USER appuser

# Expose the port the app will run on
EXPOSE 7860

# The command to start your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]