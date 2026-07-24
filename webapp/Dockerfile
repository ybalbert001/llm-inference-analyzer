FROM python:3.12-slim

WORKDIR /app
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

COPY webapp webapp

ENV PORT=8000
VOLUME /app/webapp/data
EXPOSE 8000
CMD ["python3", "webapp/app.py"]
