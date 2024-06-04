import torch
from training import *
from datas import res_data
from meta import *
from meta_learner import *
from transformers import BertModel, BertTokenizer
import Config as cg


# tokenizer = BertTokenizer.from_pretrained(cg.model_path, do_lower_case = True)
# train = MetaTask(train_examples, num_task = 13, k_support=10, k_query=2, tokenizer = tokenizer)

def strat_training(train_examples,tokenizer,test):

    args = TrainingArgs()

    learner = Learner(args)

    global_step = 0
    best_model_accuracy = 0.0  

    for epoch in range(args.meta_epoch):
    
        train = MetaTask(train_examples, num_task = cg.train_num_task, k_support=cg.k_support, k_query=cg.k_query, tokenizer = tokenizer)
        db = create_batch_of_tasks(train, is_shuffle = True, batch_size = args.outer_batch_size)

        for step, task_batch in enumerate(db):
        
            f = open(cg.log_path, 'a')
        
            acc = learner(task_batch)
        
            print('epoch:',epoch,'Step:', step, '\ttraining Acc:', acc)
            f.write("epoch:"+str(epoch)+"  Step:"+str(step)+"  acc:"+str(acc) + '\n')
        
            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Testing Mode-----------------\n")
                db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = 1)
                acc_all_test = []

                for test_batch in db_test:
                    acc = learner(test_batch, training = False)
                    acc_all_test.append(acc)

                print('Step:', step, 'Test F1:', np.mean(acc_all_test))
                f.write('Test' + str(np.mean(acc_all_test)) + '\n')
            
                random_seed(int(time.time() % 10))
            if acc > best_model_accuracy:  
                best_model_accuracy = acc  
                best_model_path = cg.save_model_path+'epoch'+str(epoch)+'_step'+str(step).pth'
                # 保存模型 
                torch.save(learner.model, best_model_path)
        
            global_step += 1
            f.close()

def run():

    train_examples, test_examples = res_data()

    tokenizer = BertTokenizer.from_pretrained(cg.model_path, do_lower_case = True)

    test = MetaTask(test_examples, num_task = cg.test_num_task, k_support=cg.k_support, k_query=cg.k_query, tokenizer = tokenizer)

    strat_training(train_examples,tokenizer,test)


if __name__ == '__main__':
    run()
    