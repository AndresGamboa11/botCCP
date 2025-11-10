# Chatbot CCP (Render)

### Despliegue rápido
1) Sube este repo a GitHub.
2) En Render → New → Web Service → conecta el repo.
3) `Build Command`: `pip install -r requirements.txt`
4) `Start Command`: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5) Configura las variables de entorno (sección `render.yaml` o panel).
6) Sube `knowledge/CAMARA.pdf` y ejecuta:
