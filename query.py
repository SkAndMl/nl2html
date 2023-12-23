import requests
from urllib.parse import urljoin
import json

def get_response():
  query = input("Enter your query: ")
  url = '<PUBLIC-URL>/query/'
  url = urljoin(url, query)
  response = requests.get(url)
  response = json.loads(response.text)
  return response['value']

if __name__ == "__main__":
  print(get_response())