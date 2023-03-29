function [new_mix,gain,delay] = distancesimulation(original_mix,Fs,distancevector,norm)
%% Distance simulation
% Here attenuation and delay will be applied to simulate different speaker
% distance. The inverse square law is used and the source is considered to
% be a point source.
format = size(original_mix);
n = format(1,2); %number of channels

gain = ones(n,1);
delay = zeros(n,1);

%Normalizing the distance to have the minimum distance at 1m while keeping
%the relative distances intact
distancevector = (norm - min(distancevector)) + distancevector;

for i=1:n
    gain(i) = 1/(distancevector(i)^2);
    delay(i) = 1+(ceil((((distancevector(i)-1)/344))*Fs)); %delay in samples
end

new_mix = zeros(length(original_mix) + max(delay),n); % new mix with the right sample size after the delay calculation

for i=1:n
    new_mix((delay(i)):delay(i)+length(original_mix)-1, i) = [gain(i).*original_mix(:,i)]; % Here I am zero padding and applying gain to each channel individually.  
end

end

