from tqdm import tqdm
import time

sum = 0
for i in tqdm(range(1000),miniters=10,desc='epoch'):
	for j in tqdm(range(20),miniters=2,desc='chuck'):
		time.sleep(0.05)
