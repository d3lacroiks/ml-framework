FROM python:3.10-slim 
WORKDIR /app 
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
COPY api.py . 
COPY model_registry/ model_registry/ 
COPY data/ data/ 
EXPOSE 5000 
CMD ["python", "api.py"] 
