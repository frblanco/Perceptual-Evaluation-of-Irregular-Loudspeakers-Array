clc
clear all
load handel.mat

%dmixFormat = {'12chn','5chn','2chn'};
dmixFormat = {'12chn','5chn','2chn'};
%stimulusName = {'bleak-midwinter', 'ID-male', 'ironman', 'rocketman','just-the-two-of-us','rocketman'};
stimulusName = {'whats'};

addpath('inAudio__downmixes\')
addpath('matlab_functions')

reference2chn = [ 1 30 0 1; 2 -30 0 1];
reference5chn = [ 1 30 0 1; 2 -30 0 1; 3 0 0 1; 4 105 0 1;  5 -105 0 1];
reference12chn = [ 1 30 0 1; 2 -30 0 1; 3 0 0 1; 4 0 0 1;  5 135 0 1; 6 -135 0 1; 7 90 0 1; 8 -90 0 1; 9 30 28 1; 10 -30 28 1; 11 150 28 1; 12 -150 28 1];
arrayname ={'reference2chn','reference5chn','reference12chn'}
Value2 ={reference2chn,reference5chn,reference12chn}
field1 = 'name';  value1 = arrayname;
field2 = 'placement';  value2 = Value2;

reference = struct(field1,value1,field2,value2);

[placementArray] = randompos(reference);
save('placementArray.mat')
 placementIdx = '1';
 %positive = left, negative = right hemisphere
 
 
 
 [avil_file] = AVILmappingv2(placementArray,dmixFormat, stimulusName, placementIdx);

