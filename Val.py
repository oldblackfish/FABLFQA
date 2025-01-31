from Utils import *
from Model_Win5 import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SRCC


def val(valset_dir, test_scene_id, load_model_path, type='val'):
    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True
    model = torch.load(load_model_path, map_location={'cuda:0': device})
    net.load_state_dict(model['state_dict'])
    net.eval()

    label_list = []
    data_list = []
    scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxemb/ourg', 'rosemary', 'Sphynx', 'Swans_1', 'Vespa']
    # scene_list = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0',
    #               'I08R0', 'I09R0', 'I10R0', 'I11R0', 'I12R0', 'I13R0', 'I14R0']
    # scene_list = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']

    for test_scene in test_scene_id:
        image_path = valset_dir + '/' + scene_list[test_scene]
        image_list = os.listdir(image_path)
        for test_image in image_list:
            patch_path = image_path + '/' + test_image
            patch_list = os.listdir(patch_path)
            output_list = 0
            for val_patch in patch_list:
                each_patch_path = patch_path + '/' + val_patch
                with h5py.File(each_patch_path, 'r') as hf:
                    label = np.array(hf.get('score_label'))
                    data = np.array(hf.get('data'))
                    data = data / 255
                    data = np.expand_dims(data, axis=0)
                    data = np.expand_dims(data, axis=0)
                    data = torch.from_numpy(data.copy())
                    data = Variable(data).to(device)
                with torch.no_grad():
                    out_score, _ = net(data)
                output_list += out_score.cpu().numpy().item()
            label_list.append(label.item())
            data_list.append(output_list / len(patch_list))

    loss = torch.nn.MSELoss().to(device)(torch.tensor(data_list), torch.tensor(label_list))
    val_SRCC = SRCC(data_list, label_list).correlation
    print(type + ' SRCC :----    %f' % val_SRCC)
    print(type + ' LOSS :----    %f' % loss)