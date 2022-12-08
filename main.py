# from transformers.modeling_gpt2 import GPT2LMHeadModel
# from transformers.modeling_tf_gpt2 import TFGPT2LMHeadModel
# from transformers.modeling_tf_bert import TFBertForSequenceClassification
# from transformers.modeling_tf_bart import TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import convert_pytorch_checkpoint_to_tf2
from transformers import TFBertForSequenceClassification, BertForSequenceClassification
from transformers import TFBertModel, BartConfig, BertTokenizer
from datasets import load_dataset, load_metric
import tensorflow as tf
import torch
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder, signature_constants, signature_def_utils, tag_constants

# tf.saved_model.save()
# model_path = r"C:\Users\刘平生\Desktop\公司业务\query结果\所用模型\bart-12L-zh-char_pt"
model_path = r"C:\Users\刘平生\Desktop\公司业务\query结果\所用模型\bart-predict10-with2All"
a1 = BartForConditionalGeneration.from_pretrained(model_path)
a2 = TFBartForConditionalGeneration.from_pretrained(model_path, from_pt=True)
# a = BartForConditionalGeneration.from_pretrained(model_path)
a2.save_weights("test_tf.h5")
toker = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
stt = ["你们喜欢这个", "我想知道喜欢的喜"]
inputs1 = toker(stt, return_tensors='pt', padding=True)
print(inputs1['input_ids'])
zz = inputs1['input_ids'][:, :-1]
print(zz)
inputs2 = toker(stt, return_tensors='tf', padding=True)
inputz = toker.batch_encode_plus(stt, return_tensors='tf', padding=True, add_special_tokens=True)
print(inputs2['input_ids'][:, :-1])
print(inputz['input_ids'][:, :-1])
# assert 1 == 2
yy = inputs2['input_ids'][:, :-1]
# print(inputs['input_ids'])
# print(inputs['input_ids'][0])
# inputs['input_ids'][0] = inputs['input_ids'][0][:-1]
# print(inputs['input_ids'][0][:-1])
ans1 = a1.generate(zz, num_beams=1, min_length=1, max_length=4, pad_token_id=0, top_p=0.9, top_k=1, temperature=0.7,
                   bos_token_id=101,
                   eos_token_id=102,
                   decoder_start_token_id=101,
                   repetition_penalty=1.0,
                   no_repeat_ngram_size=0)
print(ans1)
ans2 = a2.generate(yy, num_beams=1, min_length=1, max_length=4, pad_token_id=0, top_p=0.9, top_k=1, temperature=0.7,
                   bos_token_id=101,
                   eos_token_id=102,
                   decoder_start_token_id=101,
                   repetition_penalty=1.0,
                   no_repeat_ngram_size=0)
print(ans1)
print(ans2)
# torch.save(a1.state_dict(), "test_pt.bin")
# b = TFBertForSequenceClassification.from_pretrained("bert-base-chinese", from_pt=True)
# b = BertForSequenceClassification.from_pretrained("bert-base-uncased")

print([toker.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in ans1])
print([toker.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in ans2])

# print(sess)
# saver = tf.compat.v1.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint())
# tf(sess)