# Use the official Python image as the base image
FROM python:3.8


# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

RUN pip3 install --upgrade pip
# Install Streamlit and other dependencies


RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "web.py"]
