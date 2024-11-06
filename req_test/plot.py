import matplotlib.pyplot as plt

async_data = [1.02, 1.84, 3.31, 4.88, 8.76]
one_by_data = [2.24, 5.66, 13.33, 23.76, 36.87]
batch_data = [0.58, 1.31, 2.89, 9.22, 16.72]

x = [20, 50, 100, 200, 300]

plt.figure(figsize=(10, 6))
plt.plot(x, async_data, marker='o', label='Async Requests')
plt.plot(x, one_by_data, marker='s', label='One-by-One Requests')
plt.plot(x, batch_data, marker='^', label='Batch Requests')

# 그래프 설정
plt.xlabel('Number of Images')
plt.ylabel('Response Time (s)')
plt.title('Performance Comparison of Different Request Methods')
plt.legend()
plt.grid(True)

# 그래프를 이미지 파일로 저장
plt.savefig('performance_comparison.png')

# 그래프 출력
plt.show()