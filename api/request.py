import requests 
# Define the URL of your Flask API 
url = 'http://127.0.0.1:5000/predict' 
 
# Define the input data(image) 
image_path = 'rice_K.JPG'

# Make the request while the file is still open
with open(image_path, "rb") as f:
    files = {"image": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

    # Handle response here while still inside the `with` block
    if response.status_code == 200:
        print("✅ Success")
        print(response.json())
    else:
        print(f"❌ API Request Failed with Status Code: {response.status_code}")
        print("Response Content:", response.text)