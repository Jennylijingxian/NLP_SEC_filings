import os
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

from CBOW_10K import *
from SEC_10K_Dataset import *
from datetime import datetime
filesFolder = '/ifs/gsb/usf_interns/Parser_Project/ParsedDocumentsFolder/10KParsed/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

losses = []
EMBEDDING_DIM = 50

data = SEC_10K_Dataset(filesFolder=filesFolder, CONTEXT_SIZE=4)
data_loader = DataLoader(data, batch_size=1000)

model = CBOW_10K(EMBEDDING_DIM, data).to(device)
early_stopping = EarlyStopping()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.NLLLoss()

torch.save(model, "models/model_init")
print("Start_epoch_date_time:", datetime.now())
epoch = 0
while True:
    total_loss = 0
    i = 0
    tot_len = len(data)
    for contexts, target in data_loader:
        i += 1
        if i % 1000 == 0:
            print("PROGRESS at:", datetime.now())
            print((i * 1000) / tot_len)
        model.zero_grad()

        log_probs = model(contexts)

        loss = loss_function(log_probs, target.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        iter_loss = loss.item()
    torch.save(model, "models/model_" + str(epoch))
    with open('loss.csv', 'a+') as loss_file:
        loss_file.write(str(epoch) + "," + str(total_loss))

    early_stopping.update_loss(iter_loss)
    if early_stopping.stop_training():
        break
    losses.append(total_loss)
    epoch += 1
