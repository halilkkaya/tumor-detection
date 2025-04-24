%% BEYİN TÜMÖRÜ SINIFLANDIRMA - MODEL EĞİTİMİ
% Bu script veri setini yükler, ön işleme yapar ve iki model eğitir:
% 1. Özel tasarım CNN ağı
% 2. MobileNetV2 transfer öğrenme modeli

%% Temizlik ve Ayarlar
clear; close all; clc;
rng(42); % Reprodüksiyon için sabit rastgele sayı
disp('Beyin Tümörü Sınıflandırma Model Eğitimi Başlatılıyor...');


%% Veri Setini Yükleme ve Kontroller
disp('Veri seti yükleniyor ve kontrol ediliyor...');

datasetPath = 'C:\Users\halil ibrahim kaya\Desktop\veri';
if ~exist(datasetPath, 'dir')
    error('Veri seti bulunamadı. Lütfen Kaggle''dan indirin ve brain_tumor_dataset klasörüne çıkarın.');
end

% Klasör yapısı ve veri varlığı kontrolü
requiredFolders = {'yes', 'no'};
for i = 1:length(requiredFolders)
    if ~exist(fullfile(datasetPath, requiredFolders{i}), 'dir')
        error('Veri seti klasör yapısı hatalı. %s klasörü bulunamadı.', requiredFolders{i});
    end
end

% Görüntü veri deposu oluştur
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Veri seti boş mu kontrol et
if isempty(imds.Files)
    error('Veri seti boş. Lütfen yes/no klasörlerine görüntüleri yerleştirin.');
end

% Sınıf isimlerini ve dağılımını göster
classNames = categories(imds.Labels);
numClasses = numel(classNames);
disp('Sınıf Dağılımı:');
countEachLabel(imds)

%% Veri Setini Bölme
disp('Veri seti bölünüyor...');
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, 'randomized');
[imdsVal, imdsTest] = splitEachLabel(imdsTemp, 0.15/0.3, 'randomized');

disp(['Eğitim görüntüleri: ', num2str(numel(imdsTrain.Files))]);
disp(['Doğrulama görüntüleri: ', num2str(numel(imdsVal.Files))]);
disp(['Test görüntüleri: ', num2str(numel(imdsTest.Files))]);

%% Görüntü Ön İşleme
inputSize = [224 224 3]; % Tüm modeller için giriş boyutu

% Veri artırma (data augmentation) - Sadece eğitim verisi için
augmenter = imageDataAugmenter( ...
    'RandRotation', [-30 30], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandScale', [0.8 1.2]);
    'RandZoom', [0.8 1.2], ...     # Zoom varyasyonu
    'RandBrightness', [0.7 1.3], ... # Parlaklık ayarı
      

% Artırılmış veri depoları oluştur
augmentedImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augmentedImdsVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

augmentedImdsTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% Farklı Optimizerlar için En İyi Modelleri Oluşturma
disp('Farklı optimizerlar için modeller oluşturuluyor...');

% Optimizer listesi
optimizers = {'sgdm', 'adam', 'rmsprop'};
sensitivityLevels = {'yuksek', 'dusuk'};

% Sonuçları saklamak için
bestModels = struct();

% Her optimizer için iki farklı hassasiyet seviyesinde model oluştur
for i = 1:length(optimizers)
    optimizer = optimizers{i};
    
    for j = 1:length(sensitivityLevels)
        sensitivity = sensitivityLevels{j};
        
        disp(['Optimizer: ', optimizer, ' - Hassasiyet: ', sensitivity]);
        
        % Hassasiyet seviyesine göre parametreleri ayarla
        if strcmp(sensitivity, 'yuksek')
            % Yüksek hassasiyet için daha karmaşık model
            filterSizes = [32, 64, 128, 256, 512];
            dropoutRate = 0.45;  % Biraz daha yüksek dropout
            learningRate = 0.0004;  % Biraz daha düşük öğrenme oranı
            miniBatchSize = 16;
            l2RegFactor = 0.0012;  % Biraz daha yüksek düzenlileştirme
            maxEpochs = 55;  % Biraz daha fazla epoch
        else
            % Düşük hassasiyet için daha basit model
            filterSizes = [16, 32, 64, 128];
            dropoutRate = 0.2;
            learningRate = 0.005;
            miniBatchSize = 32;
            l2RegFactor = 0.0005;
            maxEpochs = 30;
        end
        
        % Seçilen parametreleri göster
        disp('Seçilen parametreler:');
        disp(['Filtre boyutları: [', num2str(filterSizes), ']']);
        disp(['Dropout oranı: ', num2str(dropoutRate)]);
        disp(['Öğrenme oranı: ', num2str(learningRate)]);
        disp(['Mini-batch boyutu: ', num2str(miniBatchSize)]);
        disp(['L2 düzenlileştirme faktörü: ', num2str(l2RegFactor)]);
        
        % Ağ mimarisi oluştur
        layers = [imageInputLayer(inputSize, 'Name', 'input')];
        
        % Konvolüsyon blokları
        for k = 1:length(filterSizes)
            layers = [layers
                convolution2dLayer(3, filterSizes(k), 'Padding', 'same', 'Name', ['conv', num2str(k)])
                batchNormalizationLayer('Name', ['bn', num2str(k)])
                reluLayer('Name', ['relu', num2str(k)])
                maxPooling2dLayer(2, 'Stride', 2, 'Name', ['pool', num2str(k)])];
        end
        
        % Tam bağlantılı katmanlar
        fcSize1 = filterSizes(end) * 2;
        fcSize2 = filterSizes(end);
        
        layers = [layers
            fullyConnectedLayer(fcSize1, 'Name', 'fc1')
            reluLayer('Name', 'relu_fc1')
            dropoutLayer(dropoutRate, 'Name', 'dropout1')
            
            fullyConnectedLayer(fcSize2, 'Name', 'fc2')
            reluLayer('Name', 'relu_fc2')
            dropoutLayer(dropoutRate, 'Name', 'dropout2')
            
            fullyConnectedLayer(numClasses, 'Name', 'fc3')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')];
        
        % Optimizer'a özgü parametreler
        optimizerParams = struct();
        
        switch optimizer
            case 'sgdm'
                momentum = 0.9;
                disp(['Momentum değeri: ', num2str(momentum)]);
                optimizerParams.Momentum = momentum;
            case 'rmsprop'
                squaredDecayFactor = 0.95;
                disp(['Kare bozunma faktörü: ', num2str(squaredDecayFactor)]);
                optimizerParams.SquaredGradientDecayFactor = squaredDecayFactor;
        end
        
        % Eğitim seçenekleri - Eğitim grafiğini göstermek için 'Plots' ayarını değiştirdik
        options = trainingOptions(optimizer, ...
            'MaxEpochs', maxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'ValidationData', augmentedImdsVal, ...
            'ValidationFrequency', floor(numel(imdsTrain.Files)/miniBatchSize), ...
            'ValidationPatience', 8, ...
            'InitialLearnRate', learningRate, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.2, ...
            'LearnRateDropPeriod', 10, ...
            'L2Regularization', l2RegFactor, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'Plots', 'training-progress', ...  % Eğitim grafiğini göster
            'ExecutionEnvironment', 'auto');
        
        % Optimizer'a özgü parametreleri ekle
        if isfield(optimizerParams, 'Momentum')
            options.Momentum = optimizerParams.Momentum;
        end
        if isfield(optimizerParams, 'SquaredGradientDecayFactor')
            options.SquaredGradientDecayFactor = optimizerParams.SquaredGradientDecayFactor;
        end
        
        % Ağı eğit
        try
            disp('Model eğitimi başlıyor...');
            [net, trainInfo] = trainNetwork(augmentedImdsTrain, layers, options);
            
            % Doğrulama seti üzerinde değerlendir
            YPred = classify(net, augmentedImdsVal);
            YVal = imdsVal.Labels;
            accuracy = sum(YPred == YVal)/numel(YVal);
            
            % Sonuçları kaydet
            modelKey = [optimizer, '_', sensitivity];
            bestModels.(modelKey) = struct(...
                'Network', net, ...
                'Accuracy', accuracy, ...
                'Parameters', struct(...
                    'FilterSizes', filterSizes, ...
                    'DropoutRate', dropoutRate, ...
                    'LearningRate', learningRate, ...
                    'MiniBatchSize', miniBatchSize, ...
                    'L2RegFactor', l2RegFactor, ...
                    'Optimizer', optimizer, ...
                    'OptimizerParams', optimizerParams), ...
                'TrainingInfo', trainInfo);
            
            disp(['Doğrulama doğruluğu: ', num2str(accuracy*100), '%']);
            
            % Eğitim grafiğini kaydet
            fig = figure('Visible', 'off');
            subplot(2,1,1);
            plot(trainInfo.TrainingLoss, 'LineWidth', 2);
            hold on;
            plot(trainInfo.ValidationLoss, 'LineWidth', 2);
            hold off;
            title(['Kayıp Değerleri - ', optimizer, ' (', sensitivity, ')']);
            xlabel('Epoch');
            ylabel('Kayıp');
            legend('Eğitim', 'Doğrulama', 'Location', 'northeast');
            grid on;
            
            subplot(2,1,2);
            plot(trainInfo.TrainingAccuracy, 'LineWidth', 2);
            hold on;
            plot(trainInfo.ValidationAccuracy, 'LineWidth', 2);
            hold off;
            title(['Doğruluk Değerleri - ', optimizer, ' (', sensitivity, ')']);
            xlabel('Epoch');
            ylabel('Doğruluk (%)');
            legend('Eğitim', 'Doğrulama', 'Location', 'southeast');
            grid on;
            
            saveas(fig, ['training_plot_', optimizer, '_', sensitivity, '.png']);
            close(fig);
            
            % Modeli kaydet
            netCustom = net;
            modelFileName = ['brainTumor_', optimizer, '_', sensitivity, '.mat'];
            save(modelFileName, 'netCustom', 'classNames', 'accuracy', 'filterSizes', 'dropoutRate', 'learningRate', 'miniBatchSize', 'l2RegFactor', '-v7.3');
            disp(['Model kaydedildi: ', modelFileName]);
            
        catch ME
            disp(['Hata oluştu: ', ME.message]);
        end
        
        disp('-----------------------------------');
    end
end

% Tüm modellerin performansını karşılaştır
disp('Tüm modellerin performans karşılaştırması:');
modelKeys = fieldnames(bestModels);
accuracies = zeros(length(modelKeys), 1);

for i = 1:length(modelKeys)
    key = modelKeys{i};
    accuracies(i) = bestModels.(key).Accuracy;
    disp([key, ' - Doğrulama Doğruluğu: ', num2str(accuracies(i)*100), '%']);
end

% En iyi modeli bul
[bestAccuracy, bestIdx] = max(accuracies);
bestModelKey = modelKeys{bestIdx};
disp(['En iyi model: ', bestModelKey, ' - Doğruluk: ', num2str(bestAccuracy*100), '%']);

% En iyi modeli genel en iyi model olarak kaydet
netCustom = bestModels.(bestModelKey).Network;
bestParams = bestModels.(bestModelKey).Parameters;
save('brainTumorBestNet.mat', 'netCustom', 'classNames', 'bestParams', '-v7.3');
disp('En iyi model kaydedildi: brainTumorBestNet.mat');
disp(['En iyi doğrulama doğruluğu: ', num2str(bestAccuracy*100), '%']);
% En iyi parametreleri tablo olarak oluştur
paramNames = {'Filtre Boyutları', 'Dropout Oranı', 'Öğrenme Oranı', 'Mini-batch Boyutu', 'L2 Düzenlileştirme Faktörü'};
paramValues = {num2str(bestParams.FilterSizes), bestParams.DropoutRate, bestParams.LearningRate, bestParams.MiniBatchSize, bestParams.L2RegFactor};
paramTable = table(paramNames', paramValues', 'VariableNames', {'Parametre', 'Değer'});

% Tabloyu görselleştir
figure('Name', 'En İyi Model Parametreleri', 'Position', [100 100 500 200]);
uitable('Data', table2cell(paramTable), 'ColumnName', paramTable.Properties.VariableNames, ...
    'RowName', [], 'Units', 'Normalized', 'Position', [0 0 1 1]);

% Tüm modellerin eğitim grafiklerini tek bir figürde göster
figure('Name', 'Tüm Modellerin Doğruluk Karşılaştırması', 'Position', [100 100 800 600]);
hold on;
colors = lines(length(modelKeys));
for i = 1:length(modelKeys)
    key = modelKeys{i};
    plot(bestModels.(key).TrainingInfo.ValidationAccuracy, 'LineWidth', 2, 'Color', colors(i,:));
end
hold off;
title('Tüm Modellerin Doğrulama Doğruluğu Karşılaştırması');
xlabel('Epoch');
ylabel('Doğruluk (%)');
legend(modelKeys, 'Location', 'southeast');
grid on;


%% En İyi Model Performans Değerlendirmesi
disp('En iyi model test verisi üzerinde değerlendiriliyor...');

% Test veri seti üzerinde tahminler yap
YPred = classify(netCustom, augmentedImdsTest);
YTest = imdsTest.Labels;

% Karmaşıklık matrisi (Confusion Matrix) oluştur
figure;
cm = confusionchart(YTest, YPred);
cm.Title = 'En İyi Model Karmaşıklık Matrisi';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Doğruluk hesapla
accuracy = sum(YPred == YTest)/numel(YTest);
disp(['En İyi Model Test Doğruluğu: ', num2str(accuracy*100), '%']);

% ROC eğrisi ve AUC değeri hesaplama
[~, scores] = classify(netCustom, augmentedImdsTest);

% Her sınıf için ROC eğrisi çiz
figure;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(YTest, scores(:,i), classNames(i));
    plot(X, Y, 'LineWidth', 2);
    hold on;
    disp(['AUC değeri (', char(classNames(i)), '): ', num2str(AUC)]);
end

% ROC grafiği düzenleme
hold off;
xlabel('Yanlış Pozitif Oranı');
ylabel('Doğru Pozitif Oranı');
title('En İyi Model ROC Eğrisi');
legend(classNames, 'Location', 'Best');
grid on;

%% Transfer Öğrenme Modelleri Eğitimi
disp('Transfer öğrenme modelleri hazırlanıyor...');

% Kullanılacak transfer öğrenme modelleri
transferModels = {'mobilenetv2', 'alexnet', 'resnet50'};

% Her bir transfer modeli için eğitim
for modelIdx = 1:length(transferModels)
    modelName = transferModels{modelIdx};
    disp(['Transfer öğrenme modeli hazırlanıyor: ', modelName]);
    
    % Modeli yükle
    switch modelName
        case 'mobilenetv2'
            net = mobilenetv2;
            inputLayerName = 'input_1';
            classLayerName = 'Logits';
            outputLayerName = 'ClassificationLayer_Logits';
        case 'alexnet'
            net = alexnet;
            inputLayerName = 'data';
            classLayerName = 'fc8';
            outputLayerName = 'output';
        case 'resnet50'
            net = resnet50;
            inputLayerName = 'input_1';
            classLayerName = 'fc1000';
            outputLayerName = 'ClassificationLayer_fc1000';
    end
    
    % Ağ yapısını analiz et
    disp(['Ağ yapısı analiz ediliyor: ', modelName]);
    
    % Her optimizer ve hassasiyet seviyesi için modeli eğit
    for i = 1:length(optimizers)
        optimizer = optimizers{i};
        
        for j = 1:length(sensitivityLevels)
            sensitivity = sensitivityLevels{j};
            
            disp(['Transfer öğrenme - ', modelName, ' - Optimizer: ', optimizer, ' - Hassasiyet: ', sensitivity]);
            
            % Hassasiyet seviyesine göre parametreleri ayarla
            if strcmp(sensitivity, 'yuksek')
                learningRate = 0.0001;
                miniBatchSize = 16;
                l2RegFactor = 0.001;
                maxEpochs = 20;
                freezeLayerCount = round(0.8 * numel(net.Layers)); % Daha fazla katmanı dondur
            else
                learningRate = 0.001;
                miniBatchSize = 32;
                l2RegFactor = 0.0005;
                maxEpochs = 10;
                freezeLayerCount = round(0.6 * numel(net.Layers)); % Daha az katmanı dondur
            end
            
            % Seçilen parametreleri göster
            disp('Seçilen transfer öğrenme parametreleri:');
            disp(['Öğrenme oranı: ', num2str(learningRate)]);
            disp(['Mini-batch boyutu: ', num2str(miniBatchSize)]);
            disp(['L2 düzenlileştirme faktörü: ', num2str(l2RegFactor)]);
            disp(['Dondurulan katman sayısı: ', num2str(freezeLayerCount)]);
            
            % Transfer öğrenme için katman grafiği oluştur
            lgraph = layerGraph(net);
            
            % Son katmanları değiştir
            newLayers = [
                fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
                    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
                softmaxLayer('Name', 'new_softmax')
                classificationLayer('Name', 'new_classoutput')];
            
            % Modele göre son katmanları değiştir
            if strcmp(modelName, 'mobilenetv2')
                lgraph = replaceLayer(lgraph, classLayerName, newLayers(1));
                lgraph = replaceLayer(lgraph, outputLayerName, newLayers(3));
            elseif strcmp(modelName, 'alexnet')
                lgraph = replaceLayer(lgraph, classLayerName, newLayers(1));
                lgraph = replaceLayer(lgraph, outputLayerName, newLayers(3));
            elseif strcmp(modelName, 'resnet50')
                lgraph = replaceLayer(lgraph, classLayerName, newLayers(1));
                lgraph = replaceLayer(lgraph, outputLayerName, newLayers(3));
            end
            
            % İlk katmanları dondur (öğrenmeyi kapat)
            layers = lgraph.Layers;
            
            for k = 1:freezeLayerCount
                if isprop(layers(k), 'WeightLearnRateFactor')
                    layers(k).WeightLearnRateFactor = 0;
                    layers(k).BiasLearnRateFactor = 0;
                end
            end
            
            % Optimizer'a özgü parametreler
            optimizerParams = struct();
            
            switch optimizer
                case 'sgdm'
                    momentum = 0.9;
                    disp(['Momentum değeri: ', num2str(momentum)]);
                    optimizerParams.Momentum = momentum;
                case 'rmsprop'
                    squaredDecayFactor = 0.95;
                    disp(['Kare bozunma faktörü: ', num2str(squaredDecayFactor)]);
                    optimizerParams.SquaredGradientDecayFactor = squaredDecayFactor;
            end
            
            % Eğitim seçenekleri
            options = trainingOptions(optimizer, ...
                'MaxEpochs', maxEpochs, ...
                'MiniBatchSize', miniBatchSize, ...
                'ValidationData', augmentedImdsVal, ...
                'ValidationFrequency', floor(numel(imdsTrain.Files)/miniBatchSize), ...
                'ValidationPatience', 5, ...
                'InitialLearnRate', learningRate, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.1, ...
                'LearnRateDropPeriod', 5, ...
                'L2Regularization', l2RegFactor, ...
                'Shuffle', 'every-epoch', ...
                'Verbose', true, ...
                'Plots', 'training-progress', ...
                'ExecutionEnvironment', 'auto');
            
            % Optimizer'a özgü parametreleri ekle
            if isfield(optimizerParams, 'Momentum')
                options.Momentum = optimizerParams.Momentum;
            end
            if isfield(optimizerParams, 'SquaredGradientDecayFactor')
                options.SquaredGradientDecayFactor = optimizerParams.SquaredGradientDecayFactor;
            end
            
            % Ağı eğit
            try
                disp(['Transfer öğrenme modeli eğitimi başlıyor: ', modelName, ' - ', optimizer, ' - ', sensitivity]);
                tic;
                [trainedNet, trainInfo] = trainNetwork(augmentedImdsTrain, lgraph, options);
                trainingTime = toc;
                disp(['Eğitim süresi: ', num2str(trainingTime), ' saniye']);
                
                % Doğrulama seti üzerinde değerlendir
                YPred = classify(trainedNet, augmentedImdsVal);
                YVal = imdsVal.Labels;
                accuracy = sum(YPred == YVal)/numel(YVal);
                
                % Modeli kaydet
                modelFileName = ['brainTumor_', modelName, '_', optimizer, '_', sensitivity, '.mat'];
                save(modelFileName, 'trainedNet', 'classNames', 'accuracy', 'learningRate', 'miniBatchSize', 'l2RegFactor', 'freezeLayerCount', '-v7.3');
                disp(['Model kaydedildi: ', modelFileName]);
                disp(['Doğrulama doğruluğu: ', num2str(accuracy*100), '%']);
                
                % Eğitim grafiğini kaydet
                fig = figure('Visible', 'off');
                subplot(2,1,1);
                plot(trainInfo.TrainingLoss, 'LineWidth', 2);
                hold on;
                plot(trainInfo.ValidationLoss, 'LineWidth', 2);
                hold off;
                title(['Kayıp Değerleri - ', modelName, ' - ', optimizer, ' (', sensitivity, ')']);
                xlabel('Epoch');
                ylabel('Kayıp');
                legend('Eğitim', 'Doğrulama', 'Location', 'northeast');
                grid on;
                
                subplot(2,1,2);
                plot(trainInfo.TrainingAccuracy, 'LineWidth', 2);
                hold on;
                plot(trainInfo.ValidationAccuracy, 'LineWidth', 2);
                hold off;
                title(['Doğruluk Değerleri - ', modelName, ' - ', optimizer, ' (', sensitivity, ')']);
                xlabel('Epoch');
                ylabel('Doğruluk (%)');
                legend('Eğitim', 'Doğrulama', 'Location', 'southeast');
                grid on;
                
                saveas(fig, ['training_plot_', modelName, '_', optimizer, '_', sensitivity, '.png']);
                close(fig);
                
                % Test veri seti üzerinde değerlendir
                YPredTest = classify(trainedNet, augmentedImdsTest);
                YTest = imdsTest.Labels;
                testAccuracy = sum(YPredTest == YTest)/numel(YTest);
                disp(['Test doğruluğu: ', num2str(testAccuracy*100), '%']);
                
                % Karmaşıklık matrisi
                figure('Name', [modelName, ' - ', optimizer, ' - ', sensitivity, ' Karmaşıklık Matrisi'], 'NumberTitle', 'off');
                cm = confusionchart(YTest, YPredTest);
                cm.Title = [modelName, ' - ', optimizer, ' - ', sensitivity, ' Karmaşıklık Matrisi'];
                cm.RowSummary = 'row-normalized';
                cm.ColumnSummary = 'column-normalized';
                saveas(gcf, ['confusion_matrix_', modelName, '_', optimizer, '_', sensitivity, '.png']);
                close(gcf);
                
            catch ME
                disp(['Hata oluştu: ', ME.message]);
            end
            
            disp('-----------------------------------');
        end
    end
end

%% Tüm Modellerin Karşılaştırılması
disp('Tüm modellerin karşılaştırması yapılıyor...');

% Tüm model dosyalarını bul
modelFiles = dir('brainTumor_*.mat');
modelNames = {modelFiles.name};
modelAccuracies = zeros(length(modelNames), 1);
modelTypes = cell(length(modelNames), 1);
modelOptimizers = cell(length(modelNames), 1);
modelSensitivities = cell(length(modelNames), 1);

% Her model için test doğruluğunu hesapla
for i = 1:length(modelNames)
    try
        % Model adından bilgileri çıkar
        nameParts = split(modelNames{i}(1:end-4), '_');
        if length(nameParts) >= 3
            modelTypes{i} = nameParts{2};
            modelOptimizers{i} = nameParts{3};
            if length(nameParts) >= 4
                modelSensitivities{i} = nameParts{4};
            else
                modelSensitivities{i} = 'bilinmiyor';
            end
        else
            modelTypes{i} = 'custom';
            modelOptimizers{i} = 'bilinmiyor';
            modelSensitivities{i} = 'bilinmiyor';
        end
        
        % Modeli yükle
        load(modelNames{i});
    end  
end
        % Değişken adını kontrol et
% Son katmanları değiştir
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
        'WeightLearnRateFactor', 5, 'BiasLearnRateFactor', 5)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')];

lgraph = replaceLayer(lgraph, 'Logits', newLayers(1));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newLayers(3));

% İlk katmanları dondur (öğrenmeyi kapat)
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:130
    if isprop(layers(i), 'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Daha basit alternatif
lgraph = layerGraph(netMobile);
layers = lgraph.Layers;

% Katmanları dondur
for i = 1:130
    if isprop(layers(i), 'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Doğrudan katman değişiklikleri
lgraph = replaceLayer(lgraph, 'Logits', newLayers(1));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newLayers(3));
%% MobileNetV2 Eğitim Ayarları
optionsTL = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augmentedImdsVal, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% MobileNetV2 Eğitimi
disp('MobileNetV2 modeli eğitiliyor...');
tic;
[netMobileTrained, infoTL] = trainNetwork(augmentedImdsTrain, lgraph, optionsTL);
trainingTimeTL = toc;
disp(['MobileNetV2 eğitim süresi: ' num2str(trainingTimeTL) ' saniye']);

% Modeli ve sınıf isimlerini kaydet
save('brainTumorMobileNet.mat', 'netMobileTrained', 'classNames', '-v7.3');
disp('MobileNetV2 modeli kaydedildi.');

%% Modellerin Değerlendirilmesi
disp('Modeller test verisi üzerinde değerlendiriliyor...');

% Test verisi için gerçek etiketler
YTest = imdsTest.Labels;

% Özel CNN değerlendirme
[YPredCustom, probsCustom] = classify(netCustom, augmentedImdsTest);
accuracyCustom = mean(YPredCustom == YTest);
disp(['Özel CNN Test Doğruluğu: ', num2str(accuracyCustom * 100), '%']);

% MobileNetV2 değerlendirme
[YPredMobile, probsMobile] = classify(netMobileTrained, augmentedImdsTest);
accuracyMobile = mean(YPredMobile == YTest);
disp(['MobileNetV2 Test Doğruluğu: ', num2str(accuracyMobile * 100), '%']);

% Karışıklık matrisleri
figure('Name', 'Özel CNN Karışıklık Matrisi', 'NumberTitle', 'off');
plotconfusion(YTest, YPredCustom);

figure('Name', 'MobileNetV2 Karışıklık Matrisi', 'NumberTitle', 'off');
plotconfusion(YTest, YPredMobile);

% ROC eğrileri
figure('Name', 'ROC Eğrileri', 'NumberTitle', 'off');
legendInfo = cell(numClasses, 1);
for i = 1:numClasses
    [fpr, tpr, ~, auc] = perfcurve(YTest == classNames{i}, probsCustom(:,i), true);
    plot(fpr, tpr, 'LineWidth', 2);
    hold on;
    legendInfo{i} = [classNames{i}, ' (AUC = ', num2str(auc), ')'];
end
xlabel('Yanlış Pozitif Oranı');
ylabel('Doğru Pozitif Oranı');
title('Özel CNN ROC Eğrileri');
legend(legendInfo, 'Location', 'SouthEast');
grid on;
% Bellek temizleme
clear variables -except netCustom netMobileTrained classNames;

%% Modellerin Test Veri Seti Üzerinde Performans Değerlendirmesi
disp('Modellerin test veri seti üzerinde performans değerlendirmesi başlatılıyor...');

% Test veri setini yükle
try
    % Test veri setini hazırla
    testDataPath = 'test';
    if ~exist('imdsTest', 'var')
        imdsTest = imageDatastore(testDataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        augmentedImdsTest = augmentedImageDatastore([224 224], imdsTest);
    end
    
    % Gerçek etiketleri al
    YTest = imdsTest.Labels;
    
    % Modelleri yükle
    % Özel CNN modelini yükle
    customModelPath = 'brainTumor_sgdm_yuksek.mat';
    if exist(customModelPath, 'file')
        customModelData = load(customModelPath);
        netCustom = customModelData.netCustom;
        disp('Özel CNN modeli başarıyla yüklendi.');
    else
        disp('Özel CNN modeli bulunamadı!');
    end
    
    % Transfer öğrenme modellerini yükle
    transferModels = {'mobilenetv2'};
    optimizers = {'sgdm', 'rmsprop', 'adam'};
    sensitivityLevels = {'yuksek', 'dusuk'};
    
    % Tüm transfer öğrenme modellerini yüklemeyi dene
    for modelIdx = 1:length(transferModels)
        modelName = transferModels{modelIdx};
        for i = 1:length(optimizers)
            optimizer = optimizers{i};
            for j = 1:length(sensitivityLevels)
                sensitivity = sensitivityLevels{j};
                
                modelFileName = ['brainTumor_', modelName, '_', optimizer, '_', sensitivity, '.mat'];
                if exist(modelFileName, 'file')
                    modelData = load(modelFileName);
                    % Modeli değişkene ata (dinamik değişken adı)
                    varName = ['net', upper(modelName(1)), modelName(2:end), '_', optimizer, '_', sensitivity];
                    eval([varName, ' = modelData.trainedNet;']);
                    disp([modelFileName, ' modeli başarıyla yüklendi.']);
                end
            end
        end
    end
    
    % Diğer özel modelleri yükle
    otherModels = {'brainTumor_rmsprop_yuksek.mat', 'brainTumor_sgdm_dusuk.mat', 'brainTumor_sgdm_yuksek.mat', 'brainTumor_rmsprop_dusuk.mat', 'brainTumor_adam_dusuk.mat', 'brainTumor_adam_yuksek.mat'};
    
    % sgdm_yuksek modelini özel olarak kontrol et ve yükle
    sgdmYuksekPath = 'brainTumor_sgdm_yuksek.mat';
    if exist(sgdmYuksekPath, 'file')
        sgdmYuksekData = load(sgdmYuksekPath);
        netCustom_sgdm_yuksek = sgdmYuksekData.netCustom;
        disp('sgdm_yuksek modeli özel olarak yüklendi.');
    else
        disp('sgdm_yuksek modeli bulunamadı!');
    end
    
    for i = 1:length(otherModels)
        if exist(otherModels{i}, 'file') && ~strcmp(otherModels{i}, customModelPath)
            modelData = load(otherModels{i});
            % Optimizer ve hassasiyet bilgisini dosya adından çıkar
            [~, fileName, ~] = fileparts(otherModels{i});
            parts = split(fileName, '_');
            if length(parts) >= 3
                optimizer = parts{2};
                sensitivity = parts{3};
                varName = ['netCustom_', optimizer, '_', sensitivity];
                eval([varName, ' = modelData.netCustom;']);
                disp([otherModels{i}, ' modeli başarıyla yüklendi.']);
            end
        end
    end
    
    % MobileNetV2 modelini yükle
    mobileNetPath = 'brainTumorMobileNet.mat';
    if exist(mobileNetPath, 'file')
        mobileNetData = load(mobileNetPath);
        netMobileTrained = mobileNetData.netMobileTrained;
        disp('MobileNetV2 modeli başarıyla yüklendi.');
    else
        disp('MobileNetV2 modeli bulunamadı!');
    end
    
    % Kullanılabilir modelleri topla
    availableModels = {};
    availableModelVars = {};
    
    % Özel CNN modellerini ekle
    for i = 1:length(optimizers)
        for j = 1:length(sensitivityLevels)
            varName = ['netCustom_', optimizers{i}, '_', sensitivityLevels{j}];
            if exist(varName, 'var')
                displayName = ['Özel CNN (', upper(optimizers{i}(1)), optimizers{i}(2:end), ' ', sensitivityLevels{j}, ')'];
                availableModels{end+1} = displayName;
                availableModelVars{end+1} = varName;
            end
        end
    end
    
    % MobileNetV2 modelini ekle
    if exist('netMobileTrained', 'var')
        availableModels{end+1} = 'MobileNetV2';
        availableModelVars{end+1} = 'netMobileTrained';
    end
    
    % MobileNetV2 varyasyonlarını ekle
    for i = 1:length(optimizers)
        for j = 1:length(sensitivityLevels)
            varName = ['netMobilenetv2_', optimizers{i}, '_', sensitivityLevels{j}];
            if exist(varName, 'var')
                displayName = ['MobileNetV2 (', upper(optimizers{i}(1)), optimizers{i}(2:end), ' ', sensitivityLevels{j}, ')'];
                availableModels{end+1} = displayName;
                availableModelVars{end+1} = varName;
            end
        end
    end
    
    % Performans değerlendirmesi
    disp('Model performans değerlendirmesi başlatılıyor...');
    
    % Sonuçları saklamak için tablo oluştur
    performanceTable = table('Size', [length(availableModels), 5], ...
                            'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
                            'VariableNames', {'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'});
    
    % Her model için performans değerlendirmesi yap
    for modelIdx = 1:length(availableModels)
        modelName = availableModels{modelIdx};
        modelVar = availableModelVars{modelIdx};
        
        disp(['Değerlendiriliyor: ', modelName]);
        
        % Modeli değişkenden al
        currentModel = eval(modelVar);
        
        % Test veri seti üzerinde tahmin yap
        [YPred, scores] = classify(currentModel, augmentedImdsTest);
        
        % Karışıklık matrisi oluştur
        cm = confusionmat(YTest, YPred);
        
        % Performans metriklerini hesapla
        accuracy = sum(diag(cm)) / sum(cm(:));
        
        % Sınıf sayısını belirle
        numClasses = size(cm, 1);
        
        % Her sınıf için precision, recall ve F1 hesapla
        precision = zeros(numClasses, 1);
        recall = zeros(numClasses, 1);
        f1 = zeros(numClasses, 1);
        
        for classIdx = 1:numClasses
            TP = cm(classIdx, classIdx);
            FP = sum(cm(:, classIdx)) - TP;
            FN = sum(cm(classIdx, :)) - TP;
            
            precision(classIdx) = TP / (TP + FP);
            recall(classIdx) = TP / (TP + FN);
            f1(classIdx) = 2 * (precision(classIdx) * recall(classIdx)) / (precision(classIdx) + recall(classIdx));
        end
        
        % Ortalama değerleri hesapla
        avgPrecision = mean(precision);
        avgRecall = mean(recall);
        avgF1 = mean(f1);
        
        % Sonuçları tabloya ekle
        performanceTable(modelIdx, :) = {modelName, accuracy, avgPrecision, avgRecall, avgF1};
        
        % Karışıklık matrisini görselleştir
        figure('Name', ['Karışıklık Matrisi - ', modelName], 'NumberTitle', 'off');
        confusionchart(cm, categories(YTest));
        title(['Karışıklık Matrisi - ', modelName]);
        
        % Sonuçları yazdır
        disp(['Accuracy: ', num2str(accuracy * 100), '%']);
        disp(['Precision: ', num2str(avgPrecision * 100), '%']);
        disp(['Recall: ', num2str(avgRecall * 100), '%']);
        disp(['F1 Score: ', num2str(avgF1 * 100), '%']);
        disp('-----------------------------------');
    end
    
    % Performans tablosunu göster
    disp('Tüm modellerin performans karşılaştırması:');
    disp(performanceTable);
    
    % Performans karşılaştırma grafiği
    figure('Name', 'Model Performans Karşılaştırması', 'NumberTitle', 'off');
    
    % Tablo verilerini çıkar
    modelNames = performanceTable.Model;
    accuracyValues = performanceTable.Accuracy;
    precisionValues = performanceTable.Precision;
    recallValues = performanceTable.Recall;
    f1Values = performanceTable.F1_Score;
    
    % Çubuk grafiği oluştur
    bar([accuracyValues, precisionValues, recallValues, f1Values]);
    
    % Grafik özelliklerini ayarla
    set(gca, 'XTickLabel', modelNames);
    xtickangle(45);
    ylabel('Değer');
    title('Model Performans Karşılaştırması');
    legend({'Accuracy', 'Precision', 'Recall', 'F1 Score'});
    grid on;
    
    %% Model Parametrelerini Analiz Et
    disp('Model parametreleri analizi başlatılıyor...');
    
    % Model parametrelerini saklamak için tablo oluştur
    paramTable = table('Size', [length(availableModels), 5], ...
                      'VariableTypes', {'string', 'double', 'double', 'double', 'cell'}, ...
                      'VariableNames', {'Model', 'KatmanSayisi', 'ParametreSayisi', 'OgrenmeOrani', 'OptimizerAyarlari'});
    
    % Her model için parametreleri analiz et
    for modelIdx = 1:length(availableModels)
        modelName = availableModels{modelIdx};
        modelVar = availableModelVars{modelIdx};
        
        disp(['Model Parametreleri Analiz Ediliyor: ', modelName]);
        
        % Modeli değişkenden al
        currentModel = eval(modelVar);
        
        % Katman sayısını hesapla
        layerCount = numel(currentModel.Layers);
        
        % Parametre sayısını hesapla
        paramCount = 0;
        for i = 1:layerCount
            layer = currentModel.Layers(i);
            if isprop(layer, 'Weights') && ~isempty(layer.Weights)
                paramCount = paramCount + numel(layer.Weights);
            end
            if isprop(layer, 'Bias') && ~isempty(layer.Bias)
                paramCount = paramCount + numel(layer.Bias);
            end
        end
        
        % Öğrenme oranını ve optimizer ayarlarını bul
        learningRate = NaN;
        optimizerSettings = {};
        
        % Eğitim bilgilerini kontrol et
        if isprop(currentModel, 'TrainingInfo') && ~isempty(currentModel.TrainingInfo)
            if isprop(currentModel.TrainingInfo, 'TrainingSettings') && ~isempty(currentModel.TrainingInfo.TrainingSettings)
                if isfield(currentModel.TrainingInfo.TrainingSettings, 'InitialLearnRate')
                    learningRate = currentModel.TrainingInfo.TrainingSettings.InitialLearnRate;
                end
                
                % Optimizer ayarlarını topla
                if isfield(currentModel.TrainingInfo.TrainingSettings, 'Optimizer')
                    optimizer = currentModel.TrainingInfo.TrainingSettings.Optimizer;
                    optimizerSettings{end+1} = ['Optimizer: ', optimizer];
                    
                    % SGDM için momentum
                    if strcmpi(optimizer, 'sgdm') && isfield(currentModel.TrainingInfo.TrainingSettings, 'Momentum')
                        optimizerSettings{end+1} = ['Momentum: ', num2str(currentModel.TrainingInfo.TrainingSettings.Momentum)];
                    end
                    
                    % Adam için beta1 ve beta2
                    if strcmpi(optimizer, 'adam')
                        if isfield(currentModel.TrainingInfo.TrainingSettings, 'Beta1')
                            optimizerSettings{end+1} = ['Beta1: ', num2str(currentModel.TrainingInfo.TrainingSettings.Beta1)];
                        end
                        if isfield(currentModel.TrainingInfo.TrainingSettings, 'Beta2')
                            optimizerSettings{end+1} = ['Beta2: ', num2str(currentModel.TrainingInfo.TrainingSettings.Beta2)];
                        end
                    end
                    
                    % L2 Regularization
                    if isfield(currentModel.TrainingInfo.TrainingSettings, 'L2Regularization')
                        optimizerSettings{end+1} = ['L2 Regularization: ', num2str(currentModel.TrainingInfo.TrainingSettings.L2Regularization)];
                    end
                end
            end
        end
        
        % Sonuçları tabloya ekle
        paramTable(modelIdx, :) = {modelName, layerCount, paramCount, learningRate, {optimizerSettings}};
        
        % Sonuçları yazdır
        disp(['Katman Sayısı: ', num2str(layerCount)]);
        disp(['Parametre Sayısı: ', num2str(paramCount)]);
        if ~isnan(learningRate)
            disp(['Öğrenme Oranı: ', num2str(learningRate)]);
        else
            disp('Öğrenme Oranı: Bilinmiyor');
        end
        
        if ~isempty(optimizerSettings)
            disp('Optimizer Ayarları:');
            for i = 1:length(optimizerSettings)
                disp(['  ', optimizerSettings{i}]);
            end
        else
            disp('Optimizer Ayarları: Bilinmiyor');
        end
        
        % Katman yapısını yazdır
        disp('Katman Yapısı:');
        for i = 1:layerCount % Tüm katmanları göster
            layerInfo = currentModel.Layers(i);
            layerType = class(layerInfo);
            layerType = strrep(layerType, 'nnet.cnn.layer.', '');
            
            % Katman boyutlarını al
            if isprop(layerInfo, 'NumFilters')
                layerSize = layerInfo.NumFilters;
                sizeStr = [', Filtre Sayısı: ', num2str(layerSize)];
            elseif isprop(layerInfo, 'OutputSize')
                layerSize = layerInfo.OutputSize;
                if numel(layerSize) == 1
                    sizeStr = [', Çıkış Boyutu: ', num2str(layerSize)];
                else
                    sizeStr = [', Çıkış Boyutu: [', num2str(layerSize), ']'];
                end
            else
                sizeStr = '';
            end
            
            disp(['  ', num2str(i), '. ', layerType, sizeStr]);
        end
        
        disp('-----------------------------------');
    end
    
    % Parametre tablosunu göster
    disp('Tüm modellerin parametre karşılaştırması:');
    disp(paramTable);
    
    % Parametre karşılaştırma grafiği
    figure('Name', 'Model Parametre Karşılaştırması', 'NumberTitle', 'off');
    
    % Parametre sayılarını çıkar
    paramCounts = paramTable.ParametreSayisi;
    
    % Çubuk grafiği oluştur
    bar(paramCounts);
    
    % Grafik özelliklerini ayarla
    set(gca, 'XTickLabel', modelNames);
    xtickangle(45);
    ylabel('Parametre Sayısı');
    title('Model Parametre Sayısı Karşılaştırması');
    grid on;
    
    % Logaritmik ölçek kullan (parametre sayıları çok farklı olabilir)
    set(gca, 'YScale', 'log');
    
catch ME
    disp(['Performans değerlendirme hatası: ', ME.message]);
    disp(getReport(ME));
end
