def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms.functional as TF 
    from torch.utils.data import Dataset, DataLoader
    from torchinfo import summary
    import math
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import pandas as pd
    import os
    import random


    #------------------------------REPRODUCIBILITY--------------------#
    random.seed(153)
    np.random.seed(153)
    torch.manual_seed(153)
    torch.cuda.manual_seed(153)


    #------------------------------CHOOSING DEVICE AND SAVIING--------------------#
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    #-------------------------------HYPERPARAMETERS------------------------------#
    BATCH_SIZE = 192
    EPOCHS = 400

        
    #--------------------------------DATALOADERS:--------------------------------#
    from dataloader_pytorch import mdnet_dataset

    train_dataset = mdnet_dataset(MODE='train')
    val_dataset = mdnet_dataset(MODE='val')
    test_dataset = mdnet_dataset(MODE='test')

    TRAIN_LOADER = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    VAL_LOADER = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)
    TEST_LOADER = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    train_length = len(TRAIN_LOADER.dataset)    # Used for learning rate schedule
    print("train length= ", train_length)


    #--------------------------------MODEL:--------------------------------#
    from model_pytorch import Medianet69
    #from model_torch import am3

    model = Medianet69(numclasses=35, n_channels=16)
    #n_channels = 16
    #model = am3(4*n_channels, 4*n_channels, dilation_rate = [1,2,3])
    model = model.to(device)
    #print(summary(model, input_size=(BATCH_SIZE, 3, 224, 224)))
    print(summary(model))

    #--------------------------------LEARNING RATE AND LOSS:--------------------------------#
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # LR Schedule
    initial_lr = 0.0016
    target_lr = 0.0007
    alpha = 0.042
    cosine_target_lr = target_lr * alpha

    warmup_epochs = 40
    total_steps = math.ceil(train_length / BATCH_SIZE) * EPOCHS
    warmup_steps = math.ceil(train_length / BATCH_SIZE) * warmup_epochs
    decay_steps = total_steps - (warmup_steps * 2)

    # L2 regularization for Conv layers only:
    conv_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'conv' in name:  # Checking if the layer is a convolutional layer
            conv_params.append(param)
        else:
            other_params.append(param)


    optimizer = torch.optim.AdamW([{'params': conv_params, 'weight_decay': 0.02}, {'params': other_params, 'weight_decay': 0.01}], lr = initial_lr)
    scheduler1 = lr_warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=warmup_steps, start_factor=1 ,end_factor=(target_lr/initial_lr))

    # betas = optimizer.param_groups[0]['betas']
    # print(f"Beta1: {betas[0]}, Beta2: {betas[1]}")



    #--------------------------------TRAINING AND TESTING LOOPS:--------------------------------#
    model.train()

    # Initializing log file
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "model69_base_training.csv")
    file_exists = os.path.isfile(log_file)
    log_df = pd.DataFrame(columns=["Epoch", "Loss", "Accuracy"])
    log_df.to_csv(log_file, index=False)


    training_losses = []
    test_losses = []
    accuracy_values = []
    learning_rates = []
    best_accuracy = 0

    step = 0

    for epoch in tqdm(range(EPOCHS)):
        batch_num = 0
        mini_batch_losses = []
        for input,output in TRAIN_LOADER:
            step = step + 1
            #print(step)
            batch_num = batch_num + 1
            print(batch_num)
            
            optimizer.zero_grad()

            input = input.to(device, dtype=torch.float32)
            output = output.to(device, dtype=torch.long)

            predicted_coeff = model(input)

            mini_batch_loss = loss_fn(predicted_coeff,output)
            mini_batch_losses.append(mini_batch_loss.item())
            mini_batch_loss.backward()
            
            learning_rate = optimizer.param_groups[0]['lr']
            learning_rates.append(learning_rate)
            #print(f"Learning Rate: {learning_rate}")
            optimizer.step()
            #before_lr = optimizer.param_groups[0]["lr"]
            

            if step < warmup_steps:
                scheduler1.step()
                intermediate_lr = optimizer.param_groups[0]['lr']
            if step == warmup_steps:
                optimizer.param_groups[0]['lr'] = target_lr  
            elif step > warmup_steps:
                step_cos = min(step, decay_steps+warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step_cos / (decay_steps+warmup_steps)))
                decayed = (1 - alpha) * cosine_decay + alpha
                optimizer.param_groups[0]['lr'] = target_lr * decayed
            
        training_loss = np.mean(mini_batch_losses)
        print('\n',f"Train loss is {training_loss}")
        training_losses.append(training_loss)

        # EVALUATION:
        model.eval()

        eval_batch_num = 0
        with torch.no_grad():
            mini_batch_losses = []
            accuracy_batch_losses = []

            for input,output in TEST_LOADER:
                eval_batch_num = eval_batch_num + 1
                print(eval_batch_num)

                input = input.to(device, dtype=torch.float32)
                output = output.to(device, dtype=torch.long)

                predicted_coeff = model(input)
                _,predicted_classes = torch.max(predicted_coeff, dim=1)
                
                # Calculating Accuracy:
                correct = (predicted_classes == output).sum().item()
                total = output.size(0)
                accuracy = correct/total
                accuracy_batch_losses.append(accuracy)

                mini_batch_loss = loss_fn(predicted_coeff,output)
                mini_batch_losses.append(mini_batch_loss.item())

        test_loss = np.mean(mini_batch_losses)
        test_losses.append(test_loss)
        print('\n',f"Test loss is {test_loss}")
        
        accuracy_epoch = np.mean(accuracy_batch_losses)
        accuracy_values.append(accuracy_epoch)
        print(f"Accuracy for this epoch = {accuracy_epoch*100}%")

        # Logging epoch info
        log_data = pd.DataFrame([[epoch, test_loss, accuracy_epoch]], columns=["Epoch", "Loss", "Accuracy"])
        log_data.to_csv(log_file, mode='a', header=False, index=False)

        # Saving checkpoint
        if epoch == 1:
            best_accuracy = accuracy_epoch
        if accuracy_epoch > best_accuracy:
            best_accuracy = accuracy_epoch
            model_save_path = "checkpt_model69.pth"
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': training_loss,}, model_save_path)

    #torch.save(model.state_dict(), 'checkpt69.pth')
    print(f"Best accuracy is: {best_accuracy}")


    #--------------------------------PLOTTING LOSS:--------------------------------#
    # epochs_list = range(EPOCHS)
    # epochs_list = [x + 1 for x in epochs_list]
    # #steps_list = range(0,total_steps)
    # print("final learning rate = ", learning_rates[-1])
    # plt.figure(figsize=(15,7))
    # plt.plot(epochs_list,training_losses,color='red',label='Training Loss')
    # plt.plot(epochs_list,test_losses,color='blue',label='Test Loss')
    # plt.plot(epochs_list, accuracy_values, color = 'green', label = 'Accuracy')
    # plt.xticks(epochs_list)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()


    # #plt.plot(steps_list, learning_rates,color='red',label='Training Loss')
    # #plt.figure(figsize=(15,7))
    # #plt.plot(epochs_list,training_losses,color='red',label='Training Loss')
    # #plt.plot(epochs_list,test_losses,color='blue',label='Test Loss')
    # #plt.plot(epochs_list, accuracy_values, color = 'green', label = 'Accuracy')
    # #plt.xticks(epochs_list)
    # #plt.xlabel('Epoch')
    # #plt.ylabel('Loss')
    # #plt.legend()
    # plt.show()

if __name__ == "__main__":
    train()
