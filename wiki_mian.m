%% 数据集wiki_data，哈希码长度16
clear;clc

addpath data;%data里面存放数据集
addpath utils;% utils里面是自定义函数文件
addpath Evaluation_indicators;% Evaluation_indicators里面是评价指标函数文件

%% 加载数据文件
load 'wiki_data.mat'

%db_name={'mirflickr25k','nusData','wiki_data'};
db_name='wiki_data';
nbitset  = 16; % 哈希代码长度
rng(1);

%% ---------------------------------------------------------------------------------------------------------------------
%% 计算核矩阵
disp('-------------------------------------计算训练数据和测试数据的核矩阵-------------------------------------');
if strcmp(db_name, 'mirflickr25k')
    disp('Dataset：mirflickr25k')
    inx = randperm(size(L_tr,1),size(L_tr,1));
    XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
    XTest = I_te; YTest = T_te; LTest = L_te;

elseif strcmp(db_name, 'nusData')
    disp('Dataset：nusData')
    inx = randperm(size(L_tr,1),20000);
    XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
    XTest = I_te; YTest = T_te; LTest = L_te;

elseif strcmp(db_name, 'wiki_data')
    disp('Dataset：wiki_data')
    inx = randperm(size(L_tr,1),size(L_tr,1));
    XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
    XTest = I_te; YTest = T_te; LTest = L_te;
end
% 内核表示
[n, ~] = size(YTrain);
if strcmp(db_name, 'mirflickr25k')
    n_anchors = 2000;
    anchor_image = XTrain(randsample(n, n_anchors),:); 
    anchor_text = YTrain(randsample(n, n_anchors),:);
    X_rbf_Image_tr = RBF_fast(XTrain',anchor_image'); X_rbf_Image_te = RBF_fast(XTest',anchor_image'); 
    X_rbf_Text_tr = RBF_fast(YTrain',anchor_text');  X_rbf_Text_te = RBF_fast(YTest',anchor_text'); 

elseif strcmp(db_name, 'nusData')
    n_anchors = 2000;
    anchor_image = XTrain(randsample(n, n_anchors),:); 
    anchor_text = YTrain(randsample(n, n_anchors),:);
    X_rbf_Image_tr = RBF_fast(XTrain',anchor_image'); X_rbf_Image_te = RBF_fast(XTest',anchor_image'); 
    X_rbf_Text_tr = RBF_fast(YTrain',anchor_text');  X_rbf_Text_te = RBF_fast(YTest',anchor_text'); 

elseif strcmp(db_name, 'wiki_data')
    n_anchors = 2000;
    anchor_image = XTrain(randsample(n, n_anchors),:); 
    anchor_text = YTrain(randsample(n, n_anchors),:);
    X_rbf_Image_tr = RBF_fast(XTrain',anchor_image'); X_rbf_Image_te = RBF_fast(XTest',anchor_image'); 
    X_rbf_Text_tr = RBF_fast(YTrain',anchor_text');  X_rbf_Text_te = RBF_fast(YTest',anchor_text'); 

end


%% -------------------------------------------------------------------------------------------------------------------------
%% 分布标签
disp('-------------------------------------分布标签阶段-------------------------------------');
% 初始化训练集-图像模态和文本模态的逻辑标签矩阵,维度：5000*24；
alphes=[0.5];
for alphe=alphes
    % 标签增强阶段，得到分布标签矩阵D 5000*24
    D=Lable_enhancement(LTrain,alphe);


    run=10;% 跑的次数为10次,因为参数一样的情况下，最终结果会有误差
    for i_av=1:run
        %% -------------------------------------------------------------------------------------------------------------------------
        %% 哈希代码阶段
        disp('-------------------------------------哈希代码阶段-------------------------------------');
        % 参数设置
        lambdas = [0.5];% 第2项的超参数：𝜆
        mius = [0.5];% 第3项的超参数：𝜇
        betas  = [0.5]; % 第4项的超参数：β
        theas  = [1e-2]; % 第5项的超参数：𝜆
        gammas = [1e3];% 第6项的超参数：γ
        yitas = [1e3];% 第7项的超参数:𝜂
 
        for lambda_now=lambdas
            for miu_now=mius
                for beta_now =betas
                    for thea_now=theas
                        for gamma_now=gammas
                            for yita_now=yitas

                                % 初始化超参数参数,1en=10的n次方;1e-n=10的-n次方
                                param.alphe1 = 0.3; param.alphe2 = 0.7;% 2个模态的权重:α1，α2
                                param.lambda = lambda_now;% 第2项的超参数：𝜆
                                param.miu = miu_now;% 第3项的超参数：𝜇
                                param.beta  = beta_now; % 第4项的超参数：β
                                param.thea  = thea_now; % 第5项的超参数：𝜃
                                param.gamma = gamma_now;% 第6项的超参数：γ
                                param.yita=yita_now;% 第7项的超参数:𝜂
                                
                                % 设置最大迭代次数
                                param.iters =15; % 迭代最大次数设置为15

                                % 设置聚类数量(用于构件双语义相似度S的聚类语义)
                                param.km = [100 200 500];
                                LSet = cell(1,size(param.km,2));
                                for j = 1:size(param.km,2)
                                    [a,~] = kmeans(LTrain,param.km(j),'Distance','cosine');
                                    LSet{j} = sparse(1:size(LTrain,1),a,1);
                                    LSet{j} = full(LSet{j});
                                end
                                
                                % 开始优化,对每一次的哈希代码长度进行以下操作，此时哈希码长度数组只有1位：32bit，故此循环只运行1次
                                for bit = 1:length(nbitset) 
                                    %disp('开始进入LECAGH');
                                    param.nbits = nbitset(bit);% nbitset(bit)是获取哈希代码长度数组里面第bit个哈希码长度
                                    % 设置一层投影的维度
                                    param.l=5*param.nbits;
                                    [B_train,S,S_s,S_c,objFun] = LECAGH(X_rbf_Image_tr,X_rbf_Text_tr,D,LSet,param);
                                    %disp('结束，出LECAGH');
                                    %当迭代次数等于最大迭代次数，则判断不收敛，输出不收敛的参数组
                                    if length(objFun)==param.iters
                                        fprintf('%.0e,%.0e,%.0e,%.0e,%.0e,%.0e参数不行\n', param.lambda,param.miu,param.beta,param.thea,param.gamma,param.yita);
                                        continue; % 跳过当前循环的剩余部分，进入下一次循环
                                    end
                                    
                
                                    %% -------------------------------------------------------------------------------------------------------------------------
                                    %% 哈希函数
                                    disp('-------------------------------------哈希函数阶段-------------------------------------');
                                    %% 参数初始化
                                    r=nbitset;%获取哈希码长度
                                    I_r = eye(r); % 单位矩阵维度：r x r
                                    I_an = eye(n_anchors); %单位矩阵维度：d x d
                                    epclos  = [1e-3]; % 第2项的超参数：𝜉
                                    etas  =   [1e-3];% 第3项的超参数：𝜖
                            
                                            
                                    %% 哈希函数算法阶段
                                    for epclo_now =epclos
                                        for eta_now=etas
                                                % 初始化超参数参数,1en=10的n次方;1e-n=10的-n次方
                                                param.epclo  = epclo_now; % 第2项的超参数：𝜉
                                                param.eta  = eta_now; % 第3项的超参数：𝜖
                                                
                                                % 调用函数，计算U1,U2
                                                [U1,U2]=comp_Two_stage(S_s,param,X_rbf_Text_tr,X_rbf_Image_tr,B_train,I_r,I_an,r,n_anchors);
                                               

                                                %% -------------------------------------------------------------------------------------------------------------------------
                                                %% 计算每一组循环
                                                disp('-------------------------------------开始计算MAP值-------------------------------------');                                                
                                                qBY=sign(X_rbf_Text_te*U1');
                                                rBX=B_train;
                                                mapTI = map_rank1(qBY, rBX, LTrain, LTest);
                                                map1(i_av,1) = mapTI(end);

                                                qBX=sign(X_rbf_Image_te*U2');
                                                rBY=B_train;
                                                mapIT = map_rank1(qBX, rBY, LTrain, LTest);
                                                map1(i_av,2) = mapIT(end);
                                                fprintf('Results of round %d of experiments\n', i_av);
                                                fprintf("Text Query Image: %.4f ,  Image Query Text: %.4f\n",  map1(i_av,1),map1(i_av,2));

                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    stdTtoI = std(map1(:,1))*100;
    stdItoT = std(map1(:,2))*100;
    meanTtoI = mean(map1(:,1))*100;
    meanItoT = mean(map1(:,2))*100;
    fprintf("Average performance of Text Query Image: %.2f ± %.2f\n", meanTtoI, stdTtoI);
    fprintf("Average performance of Image Query Text: %.2f ± %.2f\n", meanItoT, stdItoT);
end


