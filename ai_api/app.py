from flask import Flask, request
from transformers import ViTForImageClassification , SwinForImageClassification, AutoModel , AutoImageProcessor
from PIL import Image
import torch.nn.functional as F
app = Flask(__name__)

#{0: 'Real', 1: 'Fake'}
#{0: 'drawings', 1: 'hentai', 2: 'neutral', 3: 'porn', 4: 'sexy'}

MODEL_DEEP_FAKE = "Wvolf/ViT_Deepfake_Detection"
MODEL_NSFW = "Falconsai/nsfw_image_detection"

deep_image_processor = AutoImageProcessor.from_pretrained(MODEL_DEEP_FAKE)
nsfw_image_processor = AutoImageProcessor.from_pretrained(MODEL_NSFW)

deep_model = ViTForImageClassification.from_pretrained(MODEL_DEEP_FAKE)
nsfw_model = ViTForImageClassification.from_pretrained(MODEL_NSFW)

deep_model.eval()
nsfw_model.eval()

print("Loading Model Complete")

@app.route('/check_image', methods=['POST'])
def check_image():
    return_dict = {"deepfake_detection" : 0, "nsfw_detection" : 0}
    
    img_deep = Image.open(request.files['image']).convert("RGB") 
    pr_deep = deep_image_processor(images=img_deep, return_tensors="pt")
    pr_nsfw = nsfw_image_processor(images=img_deep, return_tensors="pt")

    result_nsfw = nsfw_model(**pr_nsfw)
    logits_nsfw = result_nsfw.logits.squeeze()
    logits_nsfw = F.softmax(logits_nsfw).tolist()
    
    prob_nsfw = logits_nsfw[1]
    print("prob_nsfw : ", prob_nsfw)

    

    
    if prob_nsfw >= 0.2 :

        result_deep = deep_model(**pr_deep)
        logits_deep = result_deep.logits.squeeze()
        logits_deep = F.softmax(logits_deep).tolist()
        prob_fake = logits_deep[1]
    
        print("prob_fake : ", prob_fake)
        
        if prob_fake >= 0.45 :
            return_dict["deepfake_detection"] = 1
            return_dict["nsfw_detection"] = 1
            
            return return_dict
        else :
            
            return_dict["deepfake_detection"] = 0
            return_dict["nsfw_detection"] = 1
            return return_dict
    
    else :
        return return_dict

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8089)
