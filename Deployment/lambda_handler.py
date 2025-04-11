import json
import base64
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Initialize the model globally for efficiency (loaded only once per Lambda execution)
model = None

# Set the number of classes (update this to match your model)
num_classes = 2

def load_model():
    global model
    if model is None:
        # Rebuild the model architecture
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # Load your saved state_dict
        model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
        model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def lambda_function(event, context):
    print(event)
    # Ensure the model is loaded before inference
    load_model()

    # Extract the image from the event body (Base64 encoded)
    try:
        if 'body' in event:  # If image is inside 'body'
            image_data = json.loads(event['body']).get('image')
        else:  # If image is directly in the root
            image_data = event.get('image')
        
        # Decode the Base64 image
        img_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps(f"Error in processing the image: {str(e)}")
        }

    # Preprocess the image
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform inference
    try:
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1)
            pred_class = pred.item()
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'predicted_class': pred_class,
                    'class_name': 'Malignant' if pred_class == 1 else 'Benign'
                })
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error during inference: {str(e)}")
        }
