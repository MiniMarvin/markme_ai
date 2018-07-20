import requests
import json

def makePost(userdata, url):
  link = url
  data = userdata
  headers = {'content-type': 'application/json'}
  response = requests.post(link, data=json.dumps(data), headers=headers)
  print(response.status_code, response.reason)
  print(response.text)
  return response.text


payload = {
	"rota": "Ceará/São Paulo"
}

res = makePost(payload, "http://localhost:8081/getTime")