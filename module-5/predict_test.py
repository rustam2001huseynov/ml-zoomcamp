import requests

url = 'http://localhost:9696/predict'

customer = {
  "gender": "male",
  "tenure": 1,
  "monthlycharges": 175.5
}

response = requests.post(url, json=customer)
result = response.json()

print(result)