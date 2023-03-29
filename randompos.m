function [s] = randompos(reference,maxdistance)
%versio 1 = reference
%version 2 = angular distortion
%version 3 = distance distortion
%version 4 = combined angular + distance distortion
%
%
%% checking arguments
if (nargin==1)
    maxdistance=1.6;
end

k=1;
for o=1:length(reference)
    channels = size(reference(o).placement,1);
    %% array naming
    for k=k:k+3
        arrayname{k,1} = {[num2str(channels(1,1)) 'chn']};
        placement_array(k,1) = {reference(o).placement;};
    end
    k=k+1
end


i = 1;

for i=1:4:size(reference,2)*4;
    channels = size(cell2mat(placement_array(i,1)),1)
    %% version 1 = reference
    version1 = cell2mat(placement_array(i,1));
    i=i+1;
    %% version 2 = angular distortion
    version2 = cell2mat(placement_array(i,1));
    for j=1:size(version2,1)
        
        if channels == 2
            channel_zone = 180;
            step = 15;
            nstep = 12;
        elseif channels == 5
            channel_zone = 60;
            step = 15;
            nstep = 12;
        elseif channels == 12
            if version2(j,3) == 0
                channel_zone = 30;
                step = 15;
                nstep = 12;
            elseif abs(version2(j,3)) == 28
                channel_zone = 30;
                step = 30;
                nstep = 6;
            elseif abs(version2(j,3)) == 56
                channel_zone = 90;
                step = 60;
                nstep = 3;
            elseif abs(version2(j,3)) == 80
                channel_zone = 90;
                step = 180;
                nstep=2;
            else
                msg = 'elevation not available in AVIL lab';
                error(msg)
            end
        end
        
        if version2(j,2) == 0
            azimuth = wrapTo180([(version2(2,2):+step:version2(1,2))]);
        else
            
            azimuth = wrapTo180((version1(j,2)-channel_zone):step:(version1(j,2)+channel_zone));
            
        end
        rdmidx = randi([1,length(azimuth)]);
        azimuth(1,rdmidx)
        version2(j,2)=azimuth(1,rdmidx);
        
    end
    
    
    placement_array{i,1}= version2;
    %% version 3 = distance distortion
    i=i+1;
    version3 =cell2mat(placement_array(i,1));
    for j=1:1:size(version3,1)
        distance = [-maxdistance:0.4:maxdistance];
        rdmidx = randi([1,length(distance)]);
        distance(1,rdmidx);
        version3(j,4)=distance(1,rdmidx);
    end
    
    placement_array{i,1}= version3;
    
    
    %% version 4 = combined distortion
    i=i+1;
    version4 =cell2mat(placement_array(i,1));
    for j=1:1:size(version4,1)
        distance = [-maxdistance:0.4:maxdistance];
        rdmidx = randi([1,length(distance)]);
        distance(1,rdmidx);
        version4(j,4)=distance(1,rdmidx);
        
        if channels == 2
            channel_zone = 60;
            step = 15;
            nstep = 12;
        elseif channels == 5
            channel_zone = 60;
            step = 15;
            nstep = 12;
        elseif channels == 12
            if version4(j,3) == 0
                channel_zone = 30;
                step = 15;
                nstep = 12;
            elseif abs(version4(j,3)) == 28
                channel_zone = 30;
                step = 30;
                nstep = 6;
            elseif abs(version4(j,3)) == 56
                channel_zone = 90;
                step = 60;
                nstep = 3;
            elseif abs(version4(j,3)) == 80
                channel_zone = 90;
                step = 180;
                nstep=2;
            else
                msg = 'elevation not available in AVIL lab';
                error(msg)
            end
            
        end
       if version4(j,2) == 0
            azimuth = wrapTo180([(version4(2,2):+step:version4(1,2))]);
        else
            
            azimuth = wrapTo180((version1(j,2)-channel_zone):step:(version1(j,2)+channel_zone));
            
        end
        rdmidx = randi([1,length(azimuth)]);
        azimuth(1,rdmidx)
        version4(j,2)=azimuth(1,rdmidx);
        
        
        
        placement_array{i,1}= version4;
    end
end

%% Rearranging random array to match reference


% for i=1:3
%
% for j=1:size(ref_fmt,2)
%  [M,I] = min(Value2{i+1}(j,2)-ref_fmt(j,2));
% end
% end


% rearrenge channels
%     for j=1:channels(1:1)
%     if






%% Creating and Saving Array Structure

field1 = 'name';  value1 = arrayname;
field2 = 'placement';  value2 = placement_array;


s = struct(field1,value1,field2,value2);
filename= [num2str(channels(1,1)) 'chn_mp_arrays'];
save(filename,'s')


end



