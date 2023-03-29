%%
load('matlab.mat');

A=table2array(veri);
pt = cvpartition(veri.item_cnt_day,"HoldOut",0.3);

veriTrain = veri(training(pt),:);
veriTest = veri(test(pt),:);

veriTrain_X=veriTrain(:,1:4);
veriTrain_Y=veriTrain(:,5);
veriTest_X=veriTest(:,1:4);
veriTest_Y=veriTest(:,5);


dataTree = fitctree(veriTrain,"item_cnt_day");
dataFitcecoc = fitcecoc(veriTrain,"item_cnt_day");
dataKnn = fitcknn(veriTrain,"item_cnt_day");


knnmodel =fitcknn(veriTrain,"item_cnt_day",'Distance','cosine','NumNeighbors',7);
dataFitlm=fitlm(veriTrain_X_array,veriTrain_Y_array,'quadratic');

hata=loss(dataTree,veriTest);
disp("Test Hatası Tree: " + hata*100)

hata = loss(dataFitcecoc,veriTest);
disp("Test Hatası Svm: " + hata*100)

hata = loss(dataKnn,veriTest);
disp("Test Hatası Knn: " + hata*100)

%% 
%Knnmodel Confusionmatrix 
[labelKNN,scoreKNN,costKNN] = predict(dataKnn,veriTest);
[mKnn,orderKnn]=confusionmat(labelKNN,veriTest_Y_array);

figure
cmKNN = confusionchart(labelKNN,veriTest_Y_array);
cmKNN.ColumnSummary = 'column-normalized';
cmKNN.RowSummary = 'row-normalized';
cmKNN.Title = 'Knnmodel Confusion Matrix';

%%
%fitcecocmodel Confusionmatrix 
[labelSVM,scoreSVM,costSVM] = predict(dataFitcecoc,veriTest);
[mSVM,orderSVM]=confusionmat(labelSVM,veriTest_Y_array);

figure
cmSVM = confusionchart(labelSVM,veriTest_Y_array);
cmSVM.ColumnSummary = 'column-normalized';
cmSVM.RowSummary = 'row-normalized';
cmSVM.Title = 'SVM Confusion Matrix';

%%
%treemodel Confusionmatrix 
[labelDT,scoreDT,costDT] = predict(dataTree,veriTest);
[mDT,orderDT]=confusionmat(labelDT,veriTest_Y_array);

figure
cmDT = confusionchart(labelDT,veriTest_Y_array);
cmDT.ColumnSummary = 'column-normalized';
cmDT.RowSummary = 'row-normalized';
cmDT.Title = 'DataTree Confusion Matrix';



%%
% %Confusion Matrix veri hazırlanış
% A=table2array(veriTrain_X);
% B=table2array(veriTrain_Y);
% C=table2array(veriTest_Y);
% t=templateLinear();
% A=A';
% rng(1);
% D=table(Mdl1.X);
% Mdl=fitcecoc(A,B,'Learners',t,'ObservationsIn','columns');
% 
% tTree=templateTree('surrogate','on');
% tEnsemble=templateEnsemble('GentleBoost',100,tTree);
% 
% options = statset('UseParallel',true);
% Mdl1 = fitcecoc(A,B,'Coding','onevsall','Learners',tEnsemble,...
%                 'Prior','uniform','NumBins',50,'Options',options);
%           
% 
% 
% %ConfusionMatrix
% CVMdl = crossval(Mdl1,'Options',options);
% oofLabel = kfoldPredict(CVMdl,'Options',options);
% confMatrix = confusionchart(B,oofLabel,'RowSummary','total-normalized');
% confMatrix.InnerPosition = [0.10 0.12 0.85 0.85];

%%
%SVM Grafik (Dağılım şeması)
%%Y=veri.item_cnt_day;
X=veri(:,1:2);

veriTable=table2array(veri);
figure
gscatter(veriTable(:,1),veriTable(:,2),veri.item_cnt_day);
h=gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Dağılım Şeması}');
xlabel('date');
ylabel('shop__id');
legend('FontSize',14,'Location','northeastoutside');
%%
%svm multiclass
SVMModels = cell(15,1);
classes = unique(veriTrain_Y_array);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = strcmp(veriTrain_Y_array,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(veriTrain_X_array,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
%%
%%%%Perfcurve Roc eğrisinde birden fazla negatif class var . Nasıl
%%%%yapılaabilir?Classların eğrilerini tek figureda çizdirme
% [label,score,cost]=predict(dataKnn,veriTest);
% [x,y,t,auc,optrocpt,suby,subynames] = perfcurve(veriTest_Y,score,2,'negClass',[1 15],...
% 'ycrit','fpr','xcrit','tpr');
% plot(suby(:,1),x)
% hold
% plot(suby(:,2),x,'r')
% hold off
% title('Two ROC curves')





%%
%Futurelara göre önem tablosu

imp = predictorImportance(dataTree);
figure(4);
bar(imp,"green","EdgeColor","#000");
title('Feature Önem Tablosu');
ylabel('Tahmin');
xlabel('Features');
h = gca;
h.XTickLabel = dataTree.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
%%

figure(1)
gscatter(veri.date,veri.shop_id,veri.item_cnt_day);
legend('FontSize',15,'Location','northeastoutside')

xlabel('date');
ylabel('shop_id');


%%
% net = patternnet(100);
% net = train(net,simpleclusterInputs,simpleclusterTargets);
% simpleclusterOutputs = sim(net,simpleclusterInputs);
% plotroc(simpleclusterTargets,simpleclusterOutputs)
% 
% veriTrain_X_array_ters=table2array(veriTrain_X);
% veriTrain_X_array_ters=veriTrain_X_array_ters';
% veriTrain_Y_array_ters=table2array(veriTrain_Y);
% veriTrain_Y_array_ters=veriTrain_Y_array_ters';
% 
% net = patternnet(50);
% net = train(net,veriTrain_X_array_ters,veriTrain_Y_array_ters);
% veriTrain_Cikti = sim(net,veriTrain_X_array_ters);
% a=perfcurve(labelKNN,scoreKNN(:,3),'0.0272108843537415')
% 
% rocObj = perfcurve(labelKNN,scoreKNN(:,1),'0.02381');
% plot(rocObj)
%%
%Sadece KNN için çalışıyor. Class uyumsuzluğundan dolayı.
disp('_____________Multiclass demo_______________')
disp('Runing Multiclass confusionmat')
a=unique(labelKNN);
b=unique(veriTest_Y_array);
[c_matrix,Result,RefereceResult]= confusion.getMatrix(labelKNN,veriTest_Y_array);
%%
%Ağaç modelini göstermek için
view(dataTree,'Mode','text')
%%
%Performans Ölçütleri Hesaplama

mDT_Result_common = multiclass_metrics_common(mDT);
[mDTsonuc,mDTreference] = multiclass_metrics_special(mDT);

mKnn_Result_common = multiclass_metrics_common(mKnn);
[mKnnsonuc,mKnnreference] = multiclass_metrics_special(mKnn);

mSVM_Result_common = multiclass_metrics_common(mSVM);
[mSVMsonuc,mSVMreference] = multiclass_metrics_special(mSVM);



%%
