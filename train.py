import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
from models.ENet_encoder import ENet_encoder
import sys
from tqdm import tqdm

def train(FLAGS):

    # Defining the hyperparameters
    device =  FLAGS.cuda
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate
    print_every = FLAGS.print_every
    eval_every = FLAGS.eval_every
    save_every = FLAGS.save_every
    nc = FLAGS.num_classes
    wd = FLAGS.weight_decay
    ip = FLAGS.input_path_train
    lp = FLAGS.label_path_train
    ipv = FLAGS.input_path_val
    lpv = FLAGS.label_path_val


    train_mode = FLAGS.train_mode
    pretrain_model = FLAGS.pretrain_model
    cityscapes_path = FLAGS.cityscapes_path
    resume_model_path = FLAGS.resume_model_path
    print ('[INFO]Defined all the hyperparameters successfully!')

    # Get the class weights
    print ('[INFO]Starting to define the class weights...')
    if len(cityscapes_path):
        pipe = loader_cityscapes(ip, cityscapes_path, batch_size='all')
        class_weights = get_class_weights(pipe, nc, isCityscapes=True)
        #class_weights = np.array([3.03507951, 13.09507946, 4.54913664, 37.64795738, 35.78537802, 31.50943831, 45.88744201, 39.936759,
        #                          6.05101481, 31.85754823, 16.92219283, 32.07766734, 47.35907214, 11.34163794, 44.31105748, 45.81085476,
        #                          45.67260936, 48.3493813, 42.02189188])
    else:
        pipe = loader(ip, lp, batch_size='all')
        class_weights = get_class_weights(pipe, nc)
    print ('[INFO]Fetched all class weights successfully!')

    # Get an instance of the model
    if train_mode.lower() == 'encoder-decoder':
        enet = ENet(nc)
        if len(pretrain_model):
            checkpoint0 = torch.load(pretrain_model)
            pretrain_dict = checkpoint0['state_dict']
            enet_dict = enet.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in enet_dict}
            enet_dict.update(pretrain_dict)
            enet.load_state_dict(enet_dict)
            print('[INFO]Previous model Instantiated!')
    else:
        enet = ENet_encoder(nc)

    print ('[INFO]Model Instantiated!')

    # Move the model to cuda if available
    enet = enet.to(device)

    # Define the criterion and the optimizer
    if len(cityscapes_path):
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device), ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = torch.optim.Adam(enet.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True, threshold=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True,
    #                                                        threshold=0.005)
    print ('[INFO]Defined the loss function and the optimizer')

    # Training Loop starts
    print ('[INFO]Staring Training...')
    print ()

    train_losses = []
    eval_losses = []


    if len(cityscapes_path):
        # Assuming we are using the Cityscapes Dataset
        bc_train = 2975 // batch_size
        bc_eval = 500 // batch_size

        pipe = loader_cityscapes(ip, cityscapes_path, batch_size)
        eval_pipe = loader_cityscapes(ipv, cityscapes_path, batch_size)
    else:
        # Assuming we are using the CamVid Dataset
        bc_train = 367 // batch_size
        bc_eval = 101 // batch_size

        pipe = loader(ip, lp, batch_size)
        eval_pipe = loader(ipv, lpv, batch_size)

    epoch = 1
    if len(resume_model_path):
        checkpoint1 = torch.load(resume_model_path)
        epoch = checkpoint1['epochs'] + 1
        enet.load_state_dict(checkpoint1['state_dict'])

    epochs = epochs
            
    for e in range(epoch, epochs+1):
            
        train_loss = 0
        print ('-'*15,'Epoch %d' % e, '-'*15)

        enet.train()
        
        for _ in tqdm(range(bc_train)):
            X_batch, mask_batch = next(pipe)
            
            #assert (X_batch >= 0. and X_batch <= 1.0).all()
            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = enet(X_batch.float())

            loss = criterion(out, mask_batch.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            
        print ()
        train_losses.append(train_loss)
        
        if (e+1) % print_every == 0:
            print ('Epoch {}/{}...'.format(e, epochs),
                    'Loss {:6f}'.format(train_loss))

        scheduler.step(train_loss)
        
        if e % eval_every == 0:
            with torch.no_grad():
                enet.eval()
                
                eval_loss = 0
                
                for _ in tqdm(range(bc_eval)):
                    inputs, labels = next(eval_pipe)

                    inputs, labels = inputs.to(device), labels.to(device)
                    out = enet(inputs)
                    
                    loss = criterion(out, labels.long())

                    eval_loss += loss.item()

                print ()
                print ('Loss {:6f}'.format(eval_loss))
                
                eval_losses.append(eval_loss)

        if e % save_every == 0:
            checkpoint = {
                'epochs' : e,
                'state_dict' : enet.state_dict()
            }
            if train_mode.lower() == 'encoder-decoder':
                torch.save(checkpoint,
                           './logs/ckpt-enet-{}-{}-{}.pth'.format(e, optimizer.state_dict()['param_groups'][0]['lr'],
                                                                  train_loss))
            else:
                torch.save(checkpoint,
                           './logs/ckpt-enet_encoder-{}-{}-{}.pth'.format(e, optimizer.state_dict()['param_groups'][0]['lr'],
                                                                          train_loss))
            print ('Model saved!')

        print ('Epoch {}/{}...'.format(e+1, epochs),
               'Total Mean Loss: {:6f}'.format(sum(train_losses) / epochs))

    print ('[INFO]Training Process complete!')
