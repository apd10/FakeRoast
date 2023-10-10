import torchvision
from FakeRoastUtil_v2 import *
import pdb


def test1(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model, module_limit_size=25000)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE, module_limit_size=25000)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)

def test2(sparsity=0.5):
    model = torchvision.models.AlexNet()
    import pdb
    pdb.set_trace()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoaster(model, False, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)
    af = get_module_params(model)
    print(int(roastable * sparsity), (af-fixed))
    assert(abs( int(roastable * sparsity) - (af - fixed)) < 5)


def test3(sparsity=0.5):
    model = torchvision.models.AlexNet()
    print(model)
    bef = get_module_params(model)
    pf = ModelRoastableParameters(model)
    s = pf.process()
    roastable = s['roastable']
    total = s['all']
    fixed = total - roastable
    
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE)
    model = roaster.process()
    print(model)


def test_mapper(sparsity=0.5):
    model = torchvision.models.AlexNet()
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 1011 }
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args)
    model = roaster.process()
    print(model)


def test_change(sparsity=0.5):
    model = torchvision.models.AlexNet()
    mapper_args = { "mapper":"pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 1011 }
    roaster = ModelRoasterGradScaler(model, True, sparsity, verbose=NONE, mapper_args=mapper_args)
    model = roaster.process()
    model.features[6].bias.data[:] = 1
    model.classifier[1].bias.data[:] = 1

    backup = copy.deepcopy(model)
    hyderator = RoastToFullModel(model)
    full_model = hyderator.process()
    print(full_model)
    backup_dict = dict(backup.named_parameters())
    backup_modules_dict = dict(backup.named_modules())
    for n,w in full_model.named_parameters():
        print( " ===== ", n, "======")
        print("full", torch.sum(w))
        if 'weight' in n and  n.replace('weight', 'WHelper.IDX') in backup_dict.keys():
            print("roast", torch.sum(backup_modules_dict[n.replace('.weight', '')].WHelper()))
        else:
            print("roast", torch.sum(backup_dict[n]))

def train_model(model, trainloader, testloader, optimizer, scheduler, metrics, DATASET, init_seed, sparse):
    device = torch.device('cuda')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    metrics['accuracy'].append(test_model(model, testloader))
    print(metrics['accuracy'][-1])
    best_acc = metrics['accuracy'][-1]
    for epoch in range(21):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()

            for p in model.parameters(): 
                if (p.requires_grad) and (p.grad is not None) and '_roast_grad_scaler' in dir(p):
                    p._roast_grad_scaler = p._roast_grad_scaler.to(p.device)
                    p.grad = p.grad / (p._roast_grad_scaler)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        # print statistics
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        metrics['post_loss'].append(running_loss/len(trainloader))
        metrics['post_accuracy'].append(test_model(model, testloader))
        print(metrics['post_accuracy'][-1])
        running_loss = 0.0
        if metrics['post_accuracy'][-1] > best_acc:
            torch.save(model.state_dict(),f"results/{DATASET}/model_post_train_{sparse}_{init_seed}.pt")
            best_acc = metrics['post_accuracy'][-1]

    return metrics

def train_and_roast(roaster, iteration, step, trainloader, testloader, DATASET, init_seed, sparse):
    device = torch.device('cuda')
    model = roaster.model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    epoch = 0
    metrics = {'loss': [], 'accuracy': [], 'roasted': [], 'post_loss': [], 'post_accuracy': []}
    metrics['accuracy'].append(test_model(model, testloader))
    print(metrics['accuracy'][-1])
    while(True):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(inputs)                
            loss = criterion(output, labels)
            loss.backward()

            for p in model.parameters(): 
                if (p.requires_grad) and (p.grad is not None) and '_roast_grad_scaler' in dir(p):
                    p._roast_grad_scaler = (roaster.aggregate_scale_partial / roaster.update_count).nan_to_num(0)
                    p.grad = (p.grad / (p._roast_grad_scaler)).nan_to_num(0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % iteration == 0:    # print every 2000 mini-batches
                if roaster.roast_array.grad is not None:
                    print("grad sum: ", torch.sum(torch.abs(roaster.roast_array.grad)))     
                metrics['loss'].append(running_loss/iteration)
                metrics['roasted'].append(roaster.k)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / iteration))
                running_loss = 0.0
                try:
                    roaster.update_k(step)    
                except:
                    print(model)
                    torch.save(model.state_dict(),f"results/{DATASET}/model_post_roast_{sparse}_{init_seed}.pt")
                    # scheduler.step()
                    return train_model(model, trainloader, testloader, optimizer, scheduler, metrics, DATASET, init_seed, sparse)
        epoch += 1
        # scheduler.step()
        metrics['accuracy'].append(test_model(model, testloader))
        print(metrics['accuracy'][-1])
        print("parameters left to be roasted: ", roaster.original_roastable_params - roaster.k - 1)

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0

    device = torch.device("cuda")
    model = model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
