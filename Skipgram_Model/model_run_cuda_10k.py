import os
# Set number of threads that is going to use
os.environ["OMP_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
os.environ["MKL_NUM_THREADS"] = "50"
os.environ["VECLIB_MAXIMUM_THREADS"] = "50"
os.environ["NUMEXPR_NUM_THREADS"] = "50"
from vocab_extractor import *
from skipgram_10k import *
from sec_10k_dataset import*

import random
from skipgram_10k import *
from sec_10k_dataset import *

# Filepaths that need to be set:
# 1. glove_path (vocab_extractor.py)
# 2. filesFolder (model_run_cuda_10k.py)
# 3. saved_input_folder (sec_10k_dataset.py)
# 4. last_model_path (optional, model_run_cuda_10k.py)

filesFolder = '/ifs/gsb/usf_interns/Parser_Project/ParsedDocumentsFoldel_10K/10K_parsed/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set hyperparameter
losses = []
EMBEDDING_DIM = 50
BATCH_SIZE = 32000
NEG = 3


def negative_sampling(targets, unigram_table, k):
    """Randomly generate negative samples from unigram table.
       Input: targets, unigran_table, # of negative samples.
       Output: nagative samples.
    """
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].data
        while len(nsample) < k:  # num of sampling
            neg = random.choice(unigram_table)
            if data.vocab2idx(neg) == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(data.prepare_sequence(nsample).view(1, -1))
    return torch.cat(neg_samples)


data = SEC_10K_Dataset(files_folder=filesFolder, context_size=4)

data_loader = DataLoader(data, batch_size=BATCH_SIZE)
unigram_table = data.unigram()

model = SkipgramNegSampling(EMBEDDING_DIM, data).to(device)
early_stopping = EarlyStopping()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # SGD or Adam??
loss_function = nn.NLLLoss()

losses = []
epoch = 0

# If continue training after pause, uncomment the following lines
# filename = '/ifs/gsb/usf_interns/test_lr/test_lr5/save_optimizer_version/models/state_2'
# def load_checkpoint(model, optimizer, losslogger, filename):
#     """Load checkpoint.
#        Input: pre-defined model, optimizer, losslogger, filename of the model. (Need to define first)
#        Output: Loaded model, optimizer, losslogger. 
#     """
#     if os.path.isfile(filename):
#         print("=> loading checkpoint '{}'".format(filename))
#         checkpoint = torch.load(filename)
#         start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         losslogger = checkpoint['losslogger']
#         print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(filename))
#     return model, optimizer, start_epoch, losslogger

# model, optimizer, epoch, losses = load_checkpoint(model=model, optimizer=optimizer, losslogger=losses, filename=filename)


print("Start_epoch_date_time:", datetime.now())


# Start Training (Only 3 epochs is for Skipgram running on scheduler(13hr/epoch & 2day limit), feel free to make changes accordingly.)
for i in range(3):# while True:
    total_loss = 0
    i = 0
    tot_len = len(data)
    for inputs, targets in data_loader:
        i += 1
        if i % 2 == 0:
            print("PROGRESS at epoch:",str(epoch),datetime.now())
            print((i * BATCH_SIZE) / tot_len)
        negs = negative_sampling(targets.to(device), unigram_table, NEG)
        model.zero_grad()
        loss = model(inputs.to(device), targets.to(device), negs.to(device))
        loss.backward()
        optimizer.step()
        iter_loss = loss.item()
        total_loss += iter_loss
        losses.append(total_loss)
    # Save parameters to dictionary
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': losses}
    torch.save(state, "models/state_" + str(epoch))
    with open('loss_test.csv', 'a+') as loss_file:
        loss_file.write(str(epoch) + "," + str(total_loss))
    early_stopping.update_loss(iter_loss)

    if early_stopping.stop_training():
        break
    losses.append(total_loss)
    epoch += 1
