import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from flask import Flask, request
from transformers import ViTForImageClassification , SwinForImageClassification, AutoModel , AutoImageProcessor
from PIL import Image
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch



app = Flask(__name__)

#{0: 'Real', 1: 'Fake'}

MODEL_DEEP_FAKE = "Wvolf/ViT_Deepfake_Detection"
MODEL_NSFW = "Falconsai/nsfw_image_detection"

deep_image_processor = AutoImageProcessor.from_pretrained(MODEL_DEEP_FAKE)
nsfw_image_processor = AutoImageProcessor.from_pretrained(MODEL_NSFW)

deep_model = ViTForImageClassification.from_pretrained(MODEL_DEEP_FAKE)
nsfw_model = ViTForImageClassification.from_pretrained(MODEL_NSFW)

deep_model.to("cuda")
nsfw_model.to("cuda")

deep_model.eval()
nsfw_model.eval()


print("Loading Model Complete")
executor = ThreadPoolExecutor(max_workers=4)  # 동시에 처리할 최대 작업 수 지정

# 실제 딥러닝 모델 함수 (예시용)
def nsfw_model_forward(pr_nsfw):
    pr_nsfw = pr_nsfw.to("cuda")
    result_nsfw = nsfw_model(**pr_nsfw)
    logits_nsfw = result_nsfw.logits.squeeze()
    logits_nsfw = F.softmax(logits_nsfw, dim=0).tolist()
    return logits_nsfw[1]  # NSFW 확률 반환

def deepfake_model_forward(pr_deep):
    pr_deep = pr_deep.to("cuda")
    result_deep = deep_model(**pr_deep)
    logits_deep = result_deep.logits.squeeze()
    logits_deep = F.softmax(logits_deep, dim=0).tolist()
    return logits_deep[1]  # Deepfake 확률 반환

def nsfw_model_multiple_forward(pr_nsfw) :
    pr_nsfw = pr_nsfw.to("cuda")
    result_nsfw = nsfw_model(**pr_nsfw)
    logits_nsfw = result_nsfw.logits.squeeze()
    logits_nsfw = F.softmax(logits_nsfw, dim=1).tolist()
    return [logits[1] for logits in logits_nsfw]

def deepfake_model_multiple_forward(pr_deep):
    pr_deep = pr_deep.to("cuda")
    result_deep = deep_model(**pr_deep)
    logits_deep = result_deep.logits.squeeze()
    logits_deep = F.softmax(logits_deep, dim=1).tolist()
    return [logits[1] for logits in logits_deep]

async def check_image():
    return_dict = {"deepfake_detection": 0, "nsfw_detection": 0}
    # 이미지 전처리
    print(request.files['image'])
    img_deep = Image.open(request.files['image']).convert("RGB")
    pr_deep = deep_image_processor(images=img_deep, return_tensors="pt")
    pr_nsfw = nsfw_image_processor(images=img_deep, return_tensors="pt")

    # NSFW 모델 실행 (비동기)
    prob_nsfw = await asyncio.get_running_loop().run_in_executor(executor, nsfw_model_forward, pr_nsfw)
    print("prob_nsfw:", prob_nsfw)

    if prob_nsfw >= 0.2:
        return_dict["nsfw_detection"] = 1
    else :
        prob_fake = await asyncio.get_running_loop().run_in_executor(executor, deepfake_model_forward, pr_deep)
        print("prob_fake:", prob_fake)
        
        if prob_fake >= 0.65:
            return_dict["deepfake_detection"] = 1
        else:
            return_dict["deepfake_detection"] = 0
            
    return return_dict

async def check_multiple_image():

    torch.cuda.empty_cache()

    return_dict = {"deepfake_detection": [], "nsfw_detection": []}
    
    # 이미지 전처리
    file_list = request.files.getlist("image")
    print(file_list)

    i = 0
    pr_deep = None
    pr_nsfw = None

    for file in file_list :
        file = Image.open(file).convert("RGB")
        print(file, i)
        deep_tensor = deep_image_processor(images=file, return_tensors="pt")
        nsfw_tensor = deep_image_processor(images=file, return_tensors="pt")
        if i == 0 :
            pr_deep = deep_tensor
            pr_nsfw = nsfw_tensor
            i = i + 1
        else :
            pr_deep['pixel_values'] = torch.cat((pr_deep['pixel_values'], deep_tensor['pixel_values']), dim = 0)
            pr_nsfw['pixel_values'] = torch.cat((pr_nsfw['pixel_values'], nsfw_tensor['pixel_values']), dim = 0)
            print(pr_deep['pixel_values'].shape)
    

    # NSFW 모델 실행 (비동기)
    prob_nsfw = await asyncio.get_running_loop().run_in_executor(executor, nsfw_model_multiple_forward, pr_nsfw)
    prob_fake = await asyncio.get_running_loop().run_in_executor(executor, deepfake_model_multiple_forward, pr_deep)
    
    print(prob_nsfw)

    for prob_nsfw_ , prob_fake_ in zip(prob_nsfw, prob_fake) :
        if prob_nsfw_ >= 0.3 :
            return_dict["nsfw_detection"].append(1)
            return_dict["deepfake_detection"].append(0)
        else :
            if prob_fake_ >= 0.65:
                return_dict["nsfw_detection"].append(0)
                return_dict["deepfake_detection"].append(1)
            else:
                return_dict["nsfw_detection"].append(0)
                return_dict["deepfake_detection"].append(0)
    
    print(return_dict)
    return return_dict

@app.route('/check_image', methods=['POST'])
async def process_image():
    result = await check_image()
    return result

@app.route('/check_multiple_image', methods=['POST'])
async def process_multiple_image() :
    result = await check_multiple_image()
    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8089)
