# import tensorflow as tf 
# import os
# # tf.debugging.set_log_device_placement(True) 


# # Explicitly place tensors on the DirectML device 

# with tf.device('/GPU:0'): 

#   a = tf.constant([1.0, 2.0, 3.0]) 

#   b = tf.constant([4.0, 5.0, 6.0]) 

# c = tf.add(a, b) 

# print(c)


import torch
import transformers
import copy
from gpt2 import utils
DEVICE = 'cuda'

# batch_data = [['1', '2', '3', '4']]
batch1 = [[['Client: I read that you should ignore them and they have to come to a conclusion that they were wrong on their own terms. Is that correct? You:'], ['It is not correct because someone who is narcissistic believes they are always right.If you ignore the person, then their thinking is that there is something wrong with you.Ignoring the person to the degree this is possible in the situation or relationship, will spare you to be misunderstood further.'], 
["Client: Sometimes, when I look at my pet cat, I think about how innocent he is and how somebody could hurt or kill him. It makes me sad because I love him, but I always think about how helpless he is. There've even been split-seconds where I felt almost tempted to kick him, followed by shame and guilt. You:"], 
['A lot of different things could be happening here. Do you feel angry or sad or anxious when you think about how helpless he is? If you have not actually kicked him, then I would encourage you to look at feelings other than guilt, since you did not hurt him. What else is there?It would probably be very helpful to talk with a therapist about the specifics of this so that you can see what else is happening for you. It could be that you feel safe with your cat, so strong emotions come up because you feel safe.']]]

batch2 = [[["Client: He wants to wear makeup and heels. He even tucks his penis away to resemble a vagina. He wants me to wear a strap on and have anal sex with him. I have tried this for him, but I don’t like it and have told him so. He keeps making comments about it and says he can't live without it. You:"], ['Depending on your own sexual history and what you grew up expecting to be "normal" in the bedroom, I can easily imagine that this came as quite a shock to you! It DOESN\'T necessarily mean, however that your husband is: gay, bisexual transgender, or even necessarily a cross-dresser etc. unless he has already told you so. I agree with the other poster who recommended you try and ask him more questions with an open and curious attitude and see if he might be open to explaining more with you. That being said, what we also know from research is that frequently what turns us on isn\'t always what we identify as.'], ["Client: I'm feeling different towards my husband. I feel I am growing from the relationship. I have been with my husband for six years and married for almost five. I just don't feel that connection anymore. I feel nothing. I don't know why or if I'm just being irrational. You:"], ["Lacey, I'm SO glad you wrote. Thousands of people are having this same feeling right now. I'm glad you're paying attention to it. When you first meet someone, there are all kinds of sparkly feelings and you both do and say lots of things to cement the attachment and create deep intimacy and connection. Then what happens is because we have that connection established, we instinctively cut back on those loving behaviours because we don't have to work hard to earn their love anymore."]]]
batch_data = [batch1, batch2]

model, tok = utils.get_model_and_tokenizer('med', transformers.AutoModelForCausalLM)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=0.001
        )

for i in range(2):
	optimizer.zero_grad()

	outer_loss_batch = []
	for task in batch_data[i]:
		inp_support, out_support, inp_query, out_query = task

		tokenized_seqs = utils.tokenize_gpt2_batch(tok, inp_support, out_support, DEVICE)
		model_copy = copy.deepcopy(model)
		inner_optimizer = torch.optim.Adam(
			params=model_copy.parameters(),
			lr=0.4
		)
		for _ in range(1):
			loss = model_copy(**tokenized_seqs).loss
			print('inner loss', loss)
			loss.backward(create_graph=True)
			inner_optimizer.step()
		inner_optimizer.zero_grad()

		tokenized_query_seqs = utils.tokenize_gpt2_batch(tok, inp_query, out_query, DEVICE)
		loss = model_copy(**tokenized_query_seqs).loss
		print('outer loss', loss)
		outer_loss_batch.append(loss)
	outer_loss = torch.mean(torch.stack(outer_loss_batch))
	outer_loss.backward()
	optimizer.step()

