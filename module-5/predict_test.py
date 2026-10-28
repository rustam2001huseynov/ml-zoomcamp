import requests

url = 'http://mlzoomcamp-churn-env.eba-sxphg2jt.eu-north-1.elasticbeanstalk.com/predict'

customer = {
  "gender": "male",
  "tenure": 1,
  "monthlycharges": 1075.5
}

response = requests.post(url, json=customer)
result = response.json()

print(result)