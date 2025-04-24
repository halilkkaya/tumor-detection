% Modelleri yükle
try
    % Tüm modeller için ortak değişkenler
    optimizers = {'sgdm', 'rmsprop', 'adam'};
    sensitivityLevels = {'yuksek', 'dusuk'};
    
    % Özel CNN modellerini yükle
    for i = 1:length(optimizers)
        for j = 1:length(sensitivityLevels)
            modelFileName = ['brainTumor_', optimizers{i}, '_', sensitivityLevels{j}, '.mat'];
            if exist(modelFileName, 'file')
                modelData = load(modelFileName);
                varName = ['netCustom_', optimizers{i}, '_', sensitivityLevels{j}];
                eval([varName, ' = modelData.netCustom;']);
                disp([modelFileName, ' modeli başarıyla yüklendi.']);
            else
                disp([modelFileName, ' modeli bulunamadı!']);
            end
        end
    end
    
    % MobileNetV2 modellerini yükle
    for i = 1:length(optimizers)
        for j = 1:length(sensitivityLevels)
            modelFileName = ['brainTumor_mobilenetv2_', optimizers{i}, '_', sensitivityLevels{j}, '.mat'];
            if exist(modelFileName, 'file')
                modelData = load(modelFileName);
                varName = ['netMobilenetv2_', optimizers{i}, '_', sensitivityLevels{j}];
                eval([varName, ' = modelData.trainedNet;']);
                disp([modelFileName, ' modeli başarıyla yüklendi.']);
            else
                disp([modelFileName, ' modeli bulunamadı!']);
            end
        end
    end
    
catch ME
    disp(['Model yükleme hatası: ', ME.message]);
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

% Eğer hiç model bulunamazsa varsayılan ekle
if isempty(availableModels)
    availableModels = {'Model bulunamadı'};
    availableModelVars = {''};
end

% Ana GUI penceresini oluştur
fig = figure('Name', 'Beyin Tümörü Tespit Sistemi', ...
    'Position', [100, 100, 800, 600], ...
    'NumberTitle', 'off', ...
    'MenuBar', 'none', ...
    'Color', [0.9 0.95 1], ... % Açık mavi arka plan
    'Resize', 'on');

% Arka plan için gradient efekti oluştur
bgAxes = axes('Parent', fig, ...
    'Units', 'normalized', ...
    'Position', [0 0 1 1], ...
    'Visible', 'off');
bgImg = ones(100, 100, 3);
% Gradient oluştur - üstten alta doğru açık maviden beyaza
for i = 1:100
    bgImg(i,:,1) = 0.8 + (i/100)*0.2; % Kırmızı kanal
    bgImg(i,:,2) = 0.85 + (i/100)*0.15; % Yeşil kanal
    bgImg(i,:,3) = 0.95 + (i/100)*0.05; % Mavi kanal
end
image(bgAxes, bgImg);
axis(bgAxes, 'off');
uistack(bgAxes, 'bottom');

% Panel oluştur
panel = uipanel(fig, 'Title', 'Beyin Tümörü Tespit Paneli', ...
    'Position', [0.05, 0.05, 0.9, 0.9], ...
    'BackgroundColor', [0.85 0.9 0.95], ... % Panel arka plan rengi
    'HighlightColor', [0.3 0.5 0.8], ... % Panel kenar rengi
    'FontWeight', 'bold', ...
    'FontSize', 12);

% Görüntü gösterme alanı
axesImg = axes('Parent', panel, ...
    'Units', 'normalized', ...
    'Position', [0.1, 0.3, 0.5, 0.6], ...
    'Box', 'on', ...
    'XColor', [0.3 0.3 0.7], ...
    'YColor', [0.3 0.3 0.7]);
title(axesImg, 'Yüklenecek Görüntü', 'FontWeight', 'bold', 'Color', [0.2 0.2 0.6]);
axis(axesImg, 'image');
axis(axesImg, 'off');

% Sonuç gösterme alanı
axesResult = axes('Parent', panel, ...
    'Units', 'normalized', ...
    'Position', [0.65, 0.3, 0.3, 0.6], ...
    'Box', 'on', ...
    'XColor', [0.3 0.3 0.7], ...
    'YColor', [0.3 0.3 0.7]);
title(axesResult, 'Sonuç', 'FontWeight', 'bold', 'Color', [0.2 0.2 0.6]);
axis(axesResult, 'off');

% Görüntü yükleme butonu
btnLoad = uicontrol(panel, 'Style', 'pushbutton', ...
    'String', 'Görüntü Yükle', ...
    'Units', 'normalized', ...
    'Position', [0.1, 0.1, 0.2, 0.1], ...
    'BackgroundColor', [0.4 0.6 0.8], ...
    'ForegroundColor', 'white', ...
    'FontWeight', 'bold', ...
    'Callback', @loadImage);

% Model seçimi
modelText = uicontrol(panel, 'Style', 'text', ...
    'String', 'Model Seçin:', ...
    'Units', 'normalized', ...
    'Position', [0.35, 0.2, 0.15, 0.05], ...
    'BackgroundColor', [0.85 0.9 0.95], ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left');

modelSelect = uicontrol(panel, 'Style', 'popupmenu', ...
    'String', availableModels, ...
    'Units', 'normalized', ...
    'Position', [0.35, 0.15, 0.2, 0.05], ...
    'BackgroundColor', [1 1 1]);

% Analiz butonu
btnAnalyze = uicontrol(panel, 'Style', 'pushbutton', ...
    'String', 'Analiz Et', ...
    'Units', 'normalized', ...
    'Position', [0.6, 0.1, 0.2, 0.1], ...
    'BackgroundColor', [0.2 0.6 0.4], ...
    'ForegroundColor', 'white', ...
    'FontWeight', 'bold', ...
    'Callback', @analyzeImage);

% Sonuç metni
resultText = uicontrol(panel, 'Style', 'text', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [0.1, 0.05, 0.8, 0.05], ...
    'BackgroundColor', [0.85 0.9 0.95], ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');

% Kemik Kadro yazısı (sağ alt köşe)
uicontrol(panel, 'Style', 'text', ...
    'String', 'Kemik Kadro', ...
    'Units', 'normalized', ...
    'Position', [0.8, 0.01, 0.15, 0.03], ...
    'BackgroundColor', [0.85 0.9 0.95], ...
    'ForegroundColor', [0.4 0.4 0.6], ...
    'FontSize', 8, ...
    'FontAngle', 'italic', ...
    'HorizontalAlignment', 'right');

% Global değişkenler
imgData = [];

% Callback fonksiyonlarını handle'lar ile tanımla
% Bu şekilde fonksiyonlar GUI bileşenlerine erişebilir
setappdata(fig, 'axesImg', axesImg);
setappdata(fig, 'axesResult', axesResult);
setappdata(fig, 'resultText', resultText);
setappdata(fig, 'modelSelect', modelSelect);
setappdata(fig, 'availableModelVars', availableModelVars);

% Görüntü yükleme fonksiyonu
function loadImage(hObject, ~)
    % GUI bileşenlerine erişim için handle'ları al
    fig = ancestor(hObject, 'figure');
    axesImg = getappdata(fig, 'axesImg');
    axesResult = getappdata(fig, 'axesResult');
    resultText = getappdata(fig, 'resultText');
    
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Görüntü Dosyaları (*.jpg, *.png, *.bmp, *.tif)'}, 'Görüntü Seçin');
    
    if isequal(filename, 0) || isequal(pathname, 0)
        return;
    end
    
    fullpath = fullfile(pathname, filename);
    imgData = imread(fullpath);
    
    % Görüntüyü göster
    axes(axesImg);
    imshow(imgData);
    title('Yüklenen Görüntü', 'FontWeight', 'bold', 'Color', [0.2 0.2 0.6]);
    
    % Sonuç alanını temizle
    axes(axesResult);
    cla;
    axis off;
    
    % Sonuç metnini temizle
    set(resultText, 'String', '');
    
    % Global değişkene ata
    setappdata(fig, 'imgData', imgData);
end

% Görüntü analiz fonksiyonu
function analyzeImage(hObject, ~)
    % GUI bileşenlerine erişim için handle'ları al
    fig = ancestor(hObject, 'figure');
    axesResult = getappdata(fig, 'axesResult');
    resultText = getappdata(fig, 'resultText');
    modelSelect = getappdata(fig, 'modelSelect');
    availableModelVars = getappdata(fig, 'availableModelVars');
    
    imgData = getappdata(fig, 'imgData');
    
    if isempty(imgData)
        set(resultText, 'String', 'Lütfen önce bir görüntü yükleyin!', 'ForegroundColor', 'red');
        return;
    end
    
    % Seçilen model
    modelIdx = get(modelSelect, 'Value');
    
    if modelIdx > length(availableModelVars) || isempty(availableModelVars{modelIdx})
        set(resultText, 'String', 'Geçerli bir model seçilmedi!', 'ForegroundColor', 'red');
        return;
    end
    
    selectedModelVar = availableModelVars{modelIdx};
    
    % Modelin var olup olmadığını kontrol et
    if ~exist(selectedModelVar, 'var')
        % Modeli dosyadan yüklemeyi dene
        modelFileName = ['brainTumor_', selectedModelVar(strfind(selectedModelVar, '_')+1:end), '.mat'];
        if exist(modelFileName, 'file')
            % Dosyadan yükle
            load(modelFileName, 'netCustom');
            selectedModel = netCustom;
        else
            set(resultText, 'String', ['Model bulunamadı: ' selectedModelVar], 'ForegroundColor', 'red');
            return;
        end
    else
        % Değişken zaten var, doğrudan kullan
        selectedModel = eval(selectedModelVar);
    end
    
    % Görüntüyü ön işleme
    img = imresize(imgData, [224 224]);
    
    % Gri tonlamalı görüntüyü RGB'ye dönüştür
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    
    % Sınıflandırma
    [label, probs] = classify(selectedModel, img);
    
    % Sonuçları göster
    probability = max(probs) * 100;
    
    if label == 'yes'
        resultColor = 'red';
        resultStr = 'TÜMÖR TESPİT EDİLDİ';
    else
        resultColor = 'green';
        resultStr = 'TÜMÖR TESPİT EDİLMEDİ';
    end
    
    % Sonuç metnini güncelle
    resultMessage = sprintf('%s (Güven: %.1f%%)', resultStr, probability);
    set(resultText, 'String', resultMessage, 'ForegroundColor', resultColor);
    
    % Sonuç görselleştirme
    axes(axesResult);
    barh(probs);
    
    % classNames değişkenini kontrol et ve kullan
    if exist('classNames', 'var')
        set(gca, 'YTick', 1:length(classNames), 'YTickLabel', classNames);
    else
        % classNames yoksa varsayılan etiketleri kullan
        defaultClassNames = {'no', 'yes'};
        set(gca, 'YTick', 1:length(defaultClassNames), 'YTickLabel', defaultClassNames);
    end
    
    xlabel('Olasılık');
    title('Sınıflandırma Sonucu', 'FontWeight', 'bold', 'Color', [0.2 0.2 0.6]);
    grid on;
end