function [array] = chn3mapping(value1, value2, value3, step)
% This miracle function will make that no two channels have the same value, and
% that the ordering of the values will follow [Left Center Right].
%
% This function works considering a wrapping of 180. Having at clockwise from
% 0 to -180.
% 

array = sort([value1 value2 value3], 'descend'); 
% array(1)= Left, array(2)=center, array(3)=right

chn = [1 3]; %Random Channels to be altered. Either Left (1) or Right (3)
opr = [1 -1];% Random Operator to be applied, to follow the direction of the channel

if abs(array(1) - array(3)) <= step
rdm_idx = randi([1 2]);
array(chn(rdm_idx)) = array(chn(rdm_idx)) + 2*step*opr(rdm_idx); % Increasing the distance between Left (1) from the Right (3) channel.
end

if abs(array(1) - array(2)) < step
rdm_idx = 1;
array(chn(rdm_idx)) = array(chn(rdm_idx)) + 2*step*opr(rdm_idx); % Increasing the distance either of the Left (1) or Center (2) channel.
end

if abs(array(2) - array(3)) < step
rdm_idx = 2;
array(chn(rdm_idx)) = array(chn(rdm_idx)) + 2*step*opr(rdm_idx); %Increasing the distance between Right (3) from the Center (2) channel.
end 



array = sort(array,'descend');
end
