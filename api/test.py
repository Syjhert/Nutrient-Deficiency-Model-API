import requests

url = "https://nutrient-deficiency-api.onrender.com/predict"

files = {'image': open("rice_K.jpg", 'rb')}
response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())