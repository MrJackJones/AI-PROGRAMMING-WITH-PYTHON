import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utility
from PIL import Image


def get_dataloders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),

        'validation': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

        'testing': transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False)
    }

    class_to_idx = image_datasets['training'].class_to_idx
    return dataloaders, class_to_idx


def get_model_from_arch(arch, hidden_units):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        classifier_input_size = model.fc.in_features
    else:
        raise RuntimeError("Unknown model")

    for param in model.parameters():
        param.requires_grad = False

    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def create_model(arch, learning_rate, hidden_units, class_to_idx):
    model = get_model_from_arch(arch, hidden_units)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()

    model.class_to_idx = class_to_idx

    return model, optimizer, criterion


def save_checkpoint(file_path, model, optimizer, arch, learning_rate, hidden_units, epochs):
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, file_path)

    print("Checkpoint Saved: '{}'".format(file_path))


def load_checkpoint(file_path, verbose=False):
    state = torch.load(file_path)

    model, optimizer, criterion = create_model(state['arch'],
                                               state['learning_rate'],
                                               state['hidden_units'],
                                               state['class_to_idx'])

    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    if verbose:
        print("Successfully loaded: {}".format(file_path))
    else:
        print("Loading checkpont failed.")

    return model


def validate(model, criterion, data_loader, use_gpu):
    model.eval()

    accuracy = 0
    test_loss = 0
    for inputs, labels in iter(data_loader):
        if use_gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)


def train(model,  criterion, optimizer, epochs, training_data_loader, validation_data_loader, use_gpu):
    model.train()
    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_data_loader):
            steps += 1

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                validation_loss, validation_accuracy = validate(model, criterion, validation_data_loader, use_gpu)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(validation_accuracy))

                running_loss = 0

                model.train()


def predict(image_path, model, use_gpu, topk=5):
    model.eval()
    image = Image.open(image_path)
    np_array = utility.process_image(image)
    tensor = torch.from_numpy(np_array)

    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True)

    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)

    probs = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]

    inverted_class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}

    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(inverted_class_to_idx[label])

    return probs.numpy()[0], mapped_classes
