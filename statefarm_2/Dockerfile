FROM python:3.9.7

RUN mkdir /code
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install uvicorn
COPY . .

# Fast API should run on port 1313.
EXPOSE 1313

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "1313"]