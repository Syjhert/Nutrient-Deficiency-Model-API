import requests

url = "https://nutrient-deficiency-model-api.onrender.com/predict"

files = {'image': open("boron.jpg", 'rb')}
response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())