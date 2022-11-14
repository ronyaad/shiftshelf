FROM python:3.8
# Create the working directory
RUN set -ex && mkdir /classifier
WORKDIR /classifier
# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
# Copy the relevant directories
# Run the web server
EXPOSE 8000
ENV PYTHONPATH /classifier
ENTRYPOINT [ "python", "/classifier/app.py" ]
# CMD python3 /classifier/app.py