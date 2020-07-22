import numpy as np 
import pandas as pd 

from mnist import *

# parameters
adversarial = True
n = 1 #inverse step size

tests = np.zeros(8)

print('Batch size 4, Epochs 10')
counter = 1
while True:
	print('Iteration', counter)
	net = SmallResNetEuler(n)
	train(net, epochs = 10, adversarial = adversarial)
	tests[0] = test(net)
	tests[1] = gaussian_test(net, 0.5)
	tests[2] = gaussian_test(net, 1.0)
	tests[3] = gaussian_test(net, 1.5)
	tests[4] = fgsm_test(net, 0.15)
	tests[5] = fgsm_test(net, 0.3)
	tests[6] = fgsm_test(net, 0.5)
	tests[7] = fgsm_test(net, 0.7)
	#tests[i, 6] = pgd_test(net, 0.3)
	#tests[i, 7] = pgd_test(net, 0.5)

	# added header in post
	if adversarial:
		pd.DataFrame([tests]).to_csv('euler' + str(n) + '_adv_data.csv', mode = 'a', header = False)
	else:
		pd.DataFrame([tests]).to_csv('euler' + str(n) + '_data.csv', mode = 'a', header = False)