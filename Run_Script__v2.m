clc
clear all
load handel.mat

addpath('data')
addpath('inAudio__downmixes\12ch_dmw-final','inAudio__64chn__downmixes','matlab_functions','data')

dmixFormat = {'12chn','5chn','2chn'};
stimulusName = {'bleak-midwinter','whats-going-on','mean-green','ironman'};



reference2chn = [ 1 30 0 1; 2 -30 0 1];
reference5chn = [ 1 30 0 1; 2 -30 0 1; 3 0 0 1; 4 105 0 1;  5 -105 0 1];
reference12chn = [ 1 30 0 1; 2 -30 0 1; 3 0 0 1; 4 0 0 1;  5 135 0 1; 6 -135 0 1; 7 90 0 1; 8 -90 0 1; 9 30 28 1; 10 -30 28 1; 11 150 28 1; 12 -150 28 1];
arrayname ={'reference2chn','reference5chn','reference12chn'}
Value2 ={reference2chn,reference5chn,reference12chn};
field1 = 'name';  value1 = arrayname;
field2 = 'placement';  value2 = Value2;

reference = struct(field1,value1,field2,value2);
maxdistance = 2;
iterations=10000;
for i=1:iterations
    [placementarray] = randompos(reference,maxdistance);
    arrayiterations{i,1} =[placementarray];
end

rdmidx = randi([1,iterations]);
placementArray = arrayiterations{rdmidx,1};
save('placementArray12.04.mat');



[avil_file] = AVILmappingv2(placementArray,dmixFormat,stimulusName);

%% Create Low anchor
fc=3500;

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
        
        
      