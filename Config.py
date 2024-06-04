model_path = 'E:/hugging_face_model_all/bert-base-uncased'

data_path = 'Amazon Review/train.json'

save_model_path = 'model_files/'

epochs = 2

train_num_task = 500

test_num_task = 5

k_support = 80

k_query = 20

num_labels = 2  

log_path = 'logs/log.txt'

meta_epoch = 10

outer_batch_size = 2

inner_batch_size = 12

outer_update_lr = 5e-5

inner_update_lr = 5e-5

inner_update_step = 10

inner_update_step_eval = 40