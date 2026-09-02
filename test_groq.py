import requests, os
env_path = "/Users/luischox/Documents/traductorKiche-Esp/traductor_kiche/.env"
with open(env_path) as f:
    for line in f:
        if line.startswith("GROQ_API_KEY="):
            key = line.strip().split("=", 1)[1].strip()

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
payload = {
    "model": "llama3-8b-8192",
    "messages": [{"role": "user", "content": "hola"}],
    "temperature": 0.7,
    "max_tokens": 512
}
resp = requests.post(url, headers=headers, json=payload)
print(resp.status_code)
print(resp.text)
