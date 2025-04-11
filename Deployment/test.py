import json
import base64
from lambda_handler import lambda_function  # assuming the main function is in lambda_function.py
from io import BytesIO
from PIL import Image

# Load an image and encode it to Base64
def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    # Path to your local image
    image_path = "C:\\Users\\ishaa\\Desktop\\OIP.jpg"

    # Encode image
    encoded_image = encode_image_to_base64(image_path)

    # Create a mock Lambda event
    event = {
        'body': json.dumps({
            'image': encoded_image
        })
    }

    # Context can be empty for local testing
    context = {}

    # Call the lambda_handler function
    response = lambda_function(event, context)

    # Print the output
    print("Response:", response)

if __name__ == '__main__':
    main()
