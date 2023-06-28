clc
clear all
load handel.mat
% Adding paths
addpath('data','inAudio__downmixes\12ch_dmw-final','inAudio__64chn__downmixes','matlab_functions','data','reference_positions')

%% Creating structure with reference layouts
reference2chn = table2array(readtable('VR-lab\2.0__reference.txt'));
reference5chn = table2array(readtable('VR-lab\5.0__reference.txt'));
reference12chn = table2array(readtable('VR-lab\7.1.4__reference.txt'));
arrayname ={'reference2chn','reference5chn','reference12chn'};
names ={reference2chn,reference5chn,reference12chn};
field1 = 'name';  value1 = arrayname;
field2 = 'placement';  value2 = names;

reference = struct(field1,value1,field2,value2);

%% 
maxdistance = 2;
placementarray = randompos_VRlab(reference,maxdistance);
save('placementArray16.04.mat');


%% Defining files and formats to render
dmixFormat = {'12chn','5chn','2chn'};
stimulusName = {'bleak-midwinter','whats-going-on','mean-green','ironman'};
[avil_file] = VRlabmapping(placementarray,dmixFormat,stimulusName);

%% Create Low anchor
fc=3500;
addpath('inAudio__downmixes\12ch_dmw-final','inAudio__64chn__downmixes','matlab_functions','data')
for i=2:5:60
    if i < 10
        name = ['00' num2str(i) '*.wav']
        file = dir(name)
    elseif i > 10 && i < 100
        name = ['0' num2str(i) '*.wav']
        file = dir(name)
    end
    filename =([file.folder '\' file.name])
    [data,fs]=audioread(filename);
    lpdata=lowpass(data,fc,fs);
    audiowrite(file.name,lpdata,fs)
end    
        
%% Images of layouts
for i=3:5:15
placementArray(i).placement(:,4) = placementArray(i).placement(:,4) + 1;
i=i+1;
placementArray(i).placement(:,4) = placementArray(i).placement(:,4) + 1;
i=i+1;
placementArray(i).placement(:,4) = placementArray(i).placement(:,4) + 1;
end    
    
    
    
[outputArg1] = formatdisp(placementArray);



      