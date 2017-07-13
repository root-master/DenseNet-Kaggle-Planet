from tqdm import tqdm
import time

sum = 0
for i in tqdm(range(1000),miniters=10,desc='epoch'):
	for j in tqdm(range(20),miniters=2,desc='chuck'):
		time.sleep(0.05)

test_slice = test_slices[-1]
X_test = load_test_data_slice(test_slice)
X_test = preprocess(X_test)
y_pred_not_aug[test_slice,:] = model.predict(X_test, batch_size=batch_size,verbose=1)
datagen.fit(X_test)
generator = datagen.flow(X_test,batch_size=batch_size)
y_pred[test_slice,:] = \
    model.predict_generator(generator, steps=len(X_test)/batch_size, verbose=1)
