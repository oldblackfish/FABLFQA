clc;close all;clear;

dataset_path = 'D:\LFIQA_Datasets\SHU\'; % Set the dataset path here
savepath = './SHU_FABLFQA_5x64x64'; % Set the save path here

% for Win5-LID dataset
load('SHU_all_info.mat');
load('SHU_all_mos.mat');
Distorted_sceneNum = 240; 

angRes = 5;             
patchsize = 64;         
stride = 64; 

inum = 1;
for iScene = 1 : Distorted_sceneNum

    label = str2num(SHU_all_mos{iScene});

    cls_name = SHU_all_info{4}{iScene};
    if strcmp(cls_name, 'P1')
        cls = 0;
    elseif strcmp(cls_name, 'P2')
        cls = 1;
    elseif strcmp(cls_name, 'P3')
        cls = 2;
    elseif strcmp(cls_name, 'P4')
        cls = 3;
    elseif strcmp(cls_name, 'P5')
        cls = 4;
    else
        fprintf('未匹配到字符串\n');
    end
    
    tic;
    idx = 1;
    h5_savedir = [savepath, '\',SHU_all_info{1}{iScene}, '\',  SHU_all_info{2}{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, SHU_all_info{6}{iScene}];
    LF = load(dataPath).im2;
    LF = LF(:,:,2:434,2:624,:);
    [U, V, ~, ~, ~] = size(LF);
    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3);
    [U, V, H, W, ~] = size(LF);
    
    for h = 1 : stride : H - patchsize + 1
        for w = 1 : stride : W - patchsize + 1    
            data = single(zeros(U * patchsize, V * patchsize));  
            for u = 1 : U
                for v = 1 : V                        
                    temp_dis = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                    temp_dis = rgb2ycbcr(temp_dis);
                    temp_dis = squeeze(temp_dis(:,:,1));   
                    data(u:angRes:U * patchsize, v:angRes:V * patchsize, :) = temp_dis;
                end
            end
%             imshow(data, []);

            img_perchannel = single(data);
            dct = single(zeros(size(img_perchannel)));
            dct_matrix = single(zeros(patchsize, patchsize, angRes*angRes));
            for i = 1:5:size(img_perchannel, 1)
                for j = 1:5:size(img_perchannel, 2)
                    dct(i:i+4, j:j+4) = dct2(img_perchannel(i:i+4, j:j+4));
                    dct_matrix((i-1)/5+1, (j-1)/5+1, :) = reshape(dct(i:i+4, j:j+4), [], 1);
                end
            end

            dct_matrix_norm = single(zeros(patchsize, patchsize, angRes*angRes));
            for i_num = 1 : angRes*angRes
                dct_matrix_norm(:,:,i_num) = normalize_to_0_255(dct_matrix(:,:,i_num));
            end
            data_DCT = dct_matrix_norm;

            SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
            h5create(SavePath_H5_name, '/data', size(data_DCT), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/data', single(data_DCT), [1,1,1], size(data_DCT));
            h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
            h5create(SavePath_H5_name, '/cls', size(cls), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/cls', single(cls), [1,1], size(cls));
            idx = idx + 1;
        end
    end
    disp(['第 ', num2str(inum), ' 个场景生成', '运行时间: ',num2str(sprintf('%.3f', toc))]);
    inum = inum + 1;
end

function normalized_data = normalize_to_0_255(data)
    % 找到数据的最小值和最大值
    min_value = min(data(:));
    max_value = max(data(:));

    % 将数据缩放到 [0, 255] 范围
    normalized_data = 255 * (data - min_value) / (max_value - min_value);

    % 将数据转换为 uint8 类型
    normalized_data = uint8(normalized_data);
end




