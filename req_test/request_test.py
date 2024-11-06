import os
import requests
import time
import asyncio
import aiohttp

# 설정
single_image_api_url = "http://203.249.8.182:8001/check_image"  # 단일 이미지 업로드 API URL
batch_image_api_url = "http://203.249.8.182:8001/check_multiple_image"  # 배치 이미지 업로드 API URL
folder_paths = ["./images/20_images", "./images/50_images" , "./images/100_images","./images/200_images","./images/300_images"]  # 이미지 폴더 경로

# 1. 하나씩 이미지를 보내는 코드 (응답 시간 측정 포함)
def send_images_one_by_one(folder_path):
    total_start_time = time.time()  # 전체 전송 시작 시간
    total_elapsed_time = 0  # 전체 소요 시간 초기화

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as img_file:
                start_time = time.time()  # 전송 시작 시간
                response = requests.post(single_image_api_url, files={'image': img_file})
                end_time = time.time()  # 응답 수신 시간
                elapsed_time = end_time - start_time
                total_elapsed_time += elapsed_time

    total_end_time = time.time()  # 전체 전송 완료 시간
    overall_elapsed_time = total_end_time - total_start_time
    print(f"Total time taken for all images: {total_elapsed_time:.2f} seconds (Measured individually)")
    print(f"Overall elapsed time: {overall_elapsed_time:.2f} seconds (From start to end)")

# 2. 여러 개의 이미지를 한 번에 보내는 코드 (응답 시간 측정 포함)
def send_images_in_batch(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            files.append(('image', (filename, open(file_path, 'rb'), 'image/jpeg')))

    if files:
        start_time = time.time()  # 전송 시작 시간
        response = requests.post(batch_image_api_url, files=files)
        end_time = time.time()  # 응답 수신 시간
        elapsed_time = end_time - start_time
        print(f"Batch sent: {response.status_code}, {response.json()}\n Time taken: {elapsed_time:.2f} seconds")

async def send_image(session, file_path):
    filename = os.path.basename(file_path)
    async with session.post(single_image_api_url, data={'image': open(file_path, 'rb')}) as response:
        start_time = time.time()
        result = await response.json()  # JSON 응답 수신
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time

# 전체 이미지 전송 함수
async def send_images_one_by_one_async(folder_path):
    total_start_time = time.time()  # 전체 전송 시작 시간
    total_elapsed_time = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_image(session, os.path.join(folder_path, filename))
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # 모든 이미지 전송 완료 대기 및 개별 응답 시간 합산
        results = await asyncio.gather(*tasks)
        total_elapsed_time = sum(results)

    total_end_time = time.time()  # 전체 전송 완료 시간
    overall_elapsed_time = total_end_time - total_start_time
    print(f"Total time taken for all images: {total_elapsed_time:.2f} seconds (Measured individually)")
    print(f"Overall elapsed time: {overall_elapsed_time:.2f} seconds (From start to end)")

# 호출 예시 (메인 이벤트 루프에서 실행)

for i in range(len(folder_paths)) :
    print(f"========== Async_one_by_one {folder_paths[i]}==========\n")
    asyncio.run(send_images_one_by_one_async(folder_paths[i]))
    print("\n")
    # 호출 예시
    print(f"========== one_by_one {folder_paths[i]} ==========\n")
    send_images_one_by_one(folder_paths[i])  # 개별 전송
    print("\n")
    print(f"========== batch {folder_paths[i]} ==========\n")
    send_images_in_batch(folder_paths[i])  # 일괄 전송
    print("\n")
