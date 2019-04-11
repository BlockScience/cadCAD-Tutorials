FROM tiangolo/uwsgi-nginx-flask:python3.6
MAINTAINER Andrew Clark - andrew@block.science
# Set the working directory to /app
COPY ./app /app
# make data directory
RUN mkdir data
RUN apt-get update
# install dependencies for textract
#RUN apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig zlib1g-dev -y 
RUN apt-get install python-dev libxml2-dev libxslt1-dev antiword poppler-utils python-pip zlib1g-dev libpulse-dev swig -y
# install packages
RUN pip install -r requirements.txt
# update packages
RUN pip install --upgrade chardet
RUN pip install --upgrade urllib3
