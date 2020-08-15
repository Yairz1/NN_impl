from NeuralNetwork import NeuralNetwork
from Utils import data_factory, show_and_save_plot
from function import *
from train import train


def train_network(data_name, batch_size, epochs, lr, momentum, L, width, patient):
    path_dic = {'Swiss': './data/NNdata/SwissRollData.mat',
                'PeaksData': './data/NNdata/PeaksData.mat',
                'GMMData': './data/NNdata/GMMData.mat'}
    C_train, C_val, X_train, X_val = data_factory(data_name, data_path=path_dic[
        data_name])  # options: 'Swiss','PeaksData','GMMData'
    n = X_val.shape[0]
    l = C_val.shape[0]

    # layer_function, activation_grad = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W,
    #                                            jacTMV_b), tanh_grad
    layer_function, activation_grad = Function(ReLU_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W,
                                               jacTMV_b), ReLU_grad

    first_layer_dim = (n, width)
    layer_dim = (width, width)
    output_dim = (width, l)
    model = NeuralNetwork(num_of_layers=L, f=layer_function,
                          activation_grad=activation_grad,  # ReLU_grad,tanh_grad
                          first_layer_dim=first_layer_dim,
                          layer_dim=layer_dim,
                          output_dim=output_dim)
    params = model.params_to_vector(model.params).copy()
    train_score, train_acc, val_score, val_acc = train(model,
                                                       model.backward,
                                                       True,  # isNeural
                                                       C_train,
                                                       C_val,
                                                       X_train,
                                                       X_val,
                                                       params,  # W0
                                                       batch_size,
                                                       epochs,
                                                       lr,
                                                       momentum=momentum, patient=patient)
    show_and_save_plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc,
                       title=f'DataSet:{data_name}| #layer = {L} | batch:{batch_size} epochs:{epochs} lr:{lr} ',
                       semilog=False)
    print(f'train_acc {train_acc}')
    print(f'val_acc {val_acc}')


batch_size = 60
epochs = 20
lr = 0.01
momentum = 0.01
L = 5
width = 20
patient = 10
train_network('PeaksData', batch_size, epochs, lr, momentum, L, width, patient)#Swiss
