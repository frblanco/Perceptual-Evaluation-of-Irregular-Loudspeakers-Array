function [placementArray] = randompos_VRlab(reference,maxdistance)
%version 1 = reference
%version 2 = low anchor
%version 3 = angular distortion
%version 4 = distance distortion
%version 5 = combined angular + distance distortion
%
%
%% checking arguments
if (nargin==1)
    maxdistance=2;
end

k=1;
for o=1:length(reference)
    channels = size(reference(o).placement,1);
    %% array naming
    for k=k:k+4
        arrayname{k,1} = {[num2str(channels(1,1)) 'chn']};
        placement_array(k,1) = {reference(o).placement;};
    end
    k=k+1;
end

i = 1;
for i=1:5:size(reference,2)*5;
    
    %% Some Init() variables for all conditions
    channels = size(cell2mat(placement_array(i,1)),1);
    
    switch channels
        case 2
            channel_zone = 30;
            step = 15;
        case 5
            channel_zone = 30;
            step = 15;
        case 12
            channel_zone = [30 90];
            step = [15 45]; % [el0, el28]
        otherwise
            msg = 'elevation not available in VRlab';
            error(msg)
    end
    
    %% version 1 = reference
    
    version1 = cell2mat(placement_array(i,1)); % Just Replicated reference. Used later for other versions, but no need to overwrite placement_array(1)
    i=i+1; %increment to next version
    
    %% version 1_lw = low anchor
  
    % maitain valuen from  placement_array(2:5:12)
    i=i+1; %increment to next version
    
    %% version 2 = angular distortion
    
    version2 = cell2mat(placement_array(i,1));    
    % loop over all channels (j) of current config
    for j=1:channels
        % check for different step for different
        %step for different elevation angle. Do this inside the j loop,
         %because in 12 channel there is different elevation with different
         %step size
         switch channels
             case 2
                  stepidx = 1;
             case 5
                  stepidx = 1;
             case 12
                 switch version2(j,3)
                     %Init() parameters
                     case 0
                         stepidx = 1;
                     case abs(30)
                         stepidx = 2;
                 end
         end
         
         
         % 1. define an array of possible speaker positions
         arrayAzimuths = ((version1(j,2)-channel_zone(stepidx)+step(stepidx)):step(stepidx):(version1(j,2)+channel_zone(stepidx)-step(stepidx)));
         
         % 2. select a random element from array
         idx = randi([1,length(arrayAzimuths)]);
         
         % 3. assign speaker position to output array
         version2(j,2) = arrayAzimuths(idx);
         
    end
    
    % sanity checks
    switch channels
        case 2
            % Chs 1 and 2 must not be equal
            % Chs 1 must be greater than Chn 2
            tmp = sort(version2(:,2), 'descend');
            version2(:,2) = tmp; % sorted in descending order, just assign (1) to Left and (2) to Right
            
        case 5
            % Chs 1,2 and 3 must not be equal
            % Ch 3 must be less than Ch 1 && greater than Ch 2
            % Delta Ch 1-2 must at least the step size
            % Delta Ch 1-3 must at least the step size
            % Delta Ch 2-3 must at least the step size
            [tmp] = chn3mapping(version2(1,2) , version2(2,2), version2(3,2), step);
            version2(1:3,2) = tmp([1 3 2]);
            
            % Ch 4 must be greater than Chn 5
            tmp = sort(version2(4:5,2), 'descend');
            version2(4:5,2) = tmp; % sorted in descending order, just assign (4) to Left Surround and (5) to Right Surround
            
            % check if there is 105 angle (not available in the VRlab)
            for j=1:5
            if abs(version2(j,2)) == 105
             op=randi([-1 1]);
             version2(j,2) = version2(j,2) + op*15;
            end
            end
        case 12
            % Chs 1,2 and 3 must not be equal
            % Ch 3 must be less than Ch 1 && greater than Ch 2
            % Delta Ch 1-2 must at least the step size
            % Delta Ch 1-3 must at least the step size
            % Delta Ch 2-3 must at least the step size
            
            [tmp] = chn3mapping(version2(1,2), version2(2,2), version2(3,2), step(stepidx));
            version2(1:3,2) = tmp([1 3 2]);
               % check if there is 105 angle (not available in the VRlab)
            for j=1:12
            if abs(version2(j,2)) == 105
             op=randi([-1 1]);
             version2(j,2) = version2(j,2) + op*15;
            end
            end
            
            if abs(version2(11,2)) == 165
                version2(11,2) = 150
            elseif abs(version2(12,2)) == 165
                version2(12,2) = -150;
            end
            % Ch 4 must be greater than Chn 5
            tmp = sort(version2(5:6,2), 'descend');
            version2(5:6,2) = tmp; % sorted in descending order, just assign (5) to Left Surround Rear and (6) to Right Surround Rear
            
            % Ch 6 must be greater than Chn 7
            tmp = sort(version2(7:8,2), 'descend');
            version2(7:8,2) = tmp; % sorted in descending order, just assign (7) to Left Surround Side and (8) to Right Surround Side
            
            % Ch 8 must be greater than Chn 9
            tmp = sort(version2(9:10,2), 'descend');
            version2(9:10,2) = tmp; % sorted in descending order, just assign (9) to Upper Left and (10) to Upper Right
            
            % Ch 10 must be greater than Chn 11
            tmp = sort(version2(11:12,2), 'descend');
            version2(11:12,2) = tmp; % sorted in descending order, just assign (11) to Upper Left Surround Rear and (12) to Upper Right Surround Near
         
        otherwise
            msg = 'Channel format may not be supported.';
            error(msg)
    end
    
    placement_array{i,1}= version2;
    i=i+1;
    
    %% version 3 = distance distortion
    
    version3 =cell2mat(placement_array(i,1));
    distance = [0:0.1:maxdistance]; %Define array of possible distances.
    for j=1:1:size(version3,1)
        %1a. Random Index to select one of the possible distances from "distance"
        idx = randi([1,length(distance)]);
        %1b. Overrite the reference value with the new one.
        version3(j,4) = distance(  idx  );
        %1c. Delete the value used above so there is no same value for the distances
        distance(idx) = []; 
    end
    placement_array{i,1}= version3;
    i=i+1;
    
    %% version 4 = combined distortion
    
    version4 =(cell2mat(placement_array(i,1)));
    distance = [0:0.1:maxdistance];
    
    % loop over all channels (j) of current config
    for j=1:channels
         % check for different step for different
         %step for different elevation angle. Do this inside the j loop,
         %because in 12 channel there is different elevation with different
         %step size
      switch channels
             case 2
                  stepidx = 1;
             case 5
                  stepidx = 1;
             case 12
                 switch version4(j,3)
                     %Init() parameters
                     case 0
                         stepidx = 1;
                     case abs(28)
                         stepidx = 2;
                     case abs(56)
                         stepidx = 3;
                     case abs(80)
                         stepidx = 4;                         
                     otherwise
                          stepidx = 1;
                 end
         end
        %1.  Angle Distortion
        
        %1a. define an array of possible speaker positions
        arrayAzimuths = ((version1(j,2)-channel_zone(stepidx)+step(stepidx)):step(stepidx):(version1(j,2)+channel_zone-step(stepidx)));
        %1b. select a random element from array
        idx = randi([1,length(arrayAzimuths)]);
        %1c. assign speaker position to output array
        version4(j,2) = arrayAzimuths(idx);
        
        %2.  Distance Distortion
        
        %2a. Random distance value
        idx = randi([1,length(distance)]); %Select a random Index to select value from "distance"
        %2b. Overwrite the reference value with the new one.
        version4(j,4) = distance(  idx  );
        %2c. Delete the value used above so there is no same value for the distances
        distance(idx) = [];
        
    end
    
    % sanity checks
    switch channels
        case 2
            % Chs 1 and 2 must not be equal
            % Chs 1 must be greater than Chn 2
            tmp = sort(version4(:,2), 'descend');
            version4(:,2) = tmp; % sorted in descending order, just assign (1) to Left and (2) to Right
            
        case 5
            % Chs 1,2 and 3 must not be equal
            % Ch 3 must be less than Ch 1 && greater than Ch 2
            % Delta Ch 1-2 must at least the step size
            % Delta Ch 1-3 must at least the step size
            % Delta Ch 2-3 must at least the step size
            [tmp] = chn3mapping(version4(1,2) , version4(2,2), version4(3,2), step(stepidx));
            version4(1:3,2) = tmp([1 3 2]);
            
            % Ch 4 must be greater than Chn 5
            tmp = sort(version4(4:5,2), 'descend');
            version4(4:5,2) = tmp; % sorted in descending order, just assign (4) to Left Surround and (5) to Right Surround
            
        case 12                  
            % Chs 1,2 and 3 must not be equal
            % Ch 3 must be less than Ch 1 && greater than Ch 2
            % Delta Ch 1-2 must at least the step size
            % Delta Ch 1-3 must at least the step size
            % Delta Ch 2-3 must at least the step size
            [tmp] = chn3mapping(version4(1,2), version4(2,2), version4(3,2), step(stepidx));
            version4(1:3,2) = tmp([1 3 2]);
            
            % Ch 4 must be greater than Chn 5
            tmp = sort(version4(5:6,2), 'descend');
            version4(5:6,2) = tmp; % sorted in descending order, just assign (5) to Left Surround Rear and (6) to Right Surround Rear
            
            % Ch 6 must be greater than Chn 7
            tmp = sort(version2(7:8,2), 'descend');
            version4(7:8,2) = tmp; % sorted in descending order, just assign (7) to Left Surround Side and (8) to Right Surround Side
            
            % Ch 8 must be greater than Chn 9
            tmp = sort(version4(9:10,2), 'descend');
            version4(9:10,2) = tmp; % sorted in descending order, just assign (9) to Upper Left and (10) to Upper Right
            
            % Ch 10 must be greater than Chn 11
            tmp = sort(version4(11:12,2), 'descend');
            version4(11:12,2) = tmp; % sorted in descending order, just assign (11) to Upper Left Surround Rear and
            %(12) to Upper Right Surround Near
            
             
            if abs(version2(11,2)) == 165
                version2(11,2) = 150
            elseif abs(version2(12,2)) == 165
                version2(12,2) = -150;
            end
        otherwise
            msg = 'Channel format may not be supported.';
            error(msg)
    end
    placement_array{i,1}= version4;
    % NO INCREMENT NEEDED; END OF LOOP!
end

%% Creating and Saving Array Structure

field1 = 'name';  value1 = arrayname;
field2 = 'placement';  value2 = placement_array;


placementArray = struct(field1,value1,field2,value2);
filename= [num2str(channels(1,1)) 'chn_mp_arrays'];
save(filename,'placementArray')



end


