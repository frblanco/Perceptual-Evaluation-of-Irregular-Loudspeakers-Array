function [avil_file] = AVILmappingv2(placementArray,dmixFormat, stimulusName, placementIdx)
%DESCRIPTION: This function creates a 64 channel .wav file from a x channel file to be used in a 64 speakers dome which is part of AVIL lab at the Technical Danish University.
%   The input is the filename that will be found in the directory and an
%   array with the channel id, azimuth, elevation and distance in the dome.
%
%
%                       chn_id     azim.[deg]  elev.[deg]  dist[m]
%    missplacementarray = [1         30             0         1;
%                          2        -30             0         0]
%
%
%
%

%% AVIL channel map
%Here we have AVIL channel map with Channel_id, Azimuth and Elevation

load('AVIL_channel_map.txt')

%% Importing audio file as an array
c=01;
for dmixIdx=1:length(dmixFormat)
    for stimulusIdx=1:length(stimulusName)
        filename = cell2mat(['\inAudio__downmixes\'  (dmixFormat(dmixIdx)) '__' (stimulusName{stimulusIdx}) '__24lufs__correct_positioned.wav']);
        [original_mix,Fs] = audioread(filename);
        if string(dmixFormat(dmixIdx)) == '12chn'
            s=9;
        elseif string(dmixFormat(dmixIdx)) == '5chn'
            s=5;
        elseif string(dmixFormat(dmixIdx)) == '5chn'
            s=1;
        end
        for h=s:1:s-1+size(placementArray,1)/3
            %Defining input format
            nchannels = size(original_mix,2);
            bassmanagement = 'of';
            
            if nchannels == 2
                msg = [stimulusName ' is a Stereo input file.'];
                disp(msg)
                
            elseif nchannels == 3
                msg = [stimulusName ' is 3.0 input file.'];
                disp(msg)
                
            elseif nchannels == 4
                msg = [stimulusName ' is a Quad input file.']';
                disp(msg)
                
            elseif nchannels == 5
                msg = [stimulusName ' is a 5.0 input file.'];
                disp(msg)
                
            elseif nchannels == 12
                msg = [stimulusName ' is a 7.1.4 input file. Bass Management will be turned on'];
                disp(msg)
                bassmanagement = 'on';
            end
            
            %% Bass Management
            %Applying Bass Management. As according the MPEG-H standard. Routing the
            %LFE to both CH_M_L030 and CH_M_R030 withouth gain or eq changes.
            if bassmanagement == 'on'
                LFE = original_mix(:,4);
                original_mix (:,1) = original_mix (:,1) + 0.7079*LFE;
                original_mix (:,2) = original_mix (:,2) + 0.7079*LFE;
            end
            
            %% Distance simulation
            % Here attenuation and delay will be applied to simulate different speaker
            % distance. The inverse square law is used and the source is considered to
            % be a point source.
            %Normalizing the distance to have the minimum distance at 1m while keeping
            %the relative distances intact
            distancevector = placementArray(h).placement(:,4);
            norm = 1; %physical distance/radius of AVIL.
            
            [new_mix,gain,delay] = distancesimulation(original_mix,Fs,distancevector,norm);
            
            avil_file = zeros(length(new_mix),64); %creating an empty 64 channel array to route the right channels.
            
            
            
            %% Flexible Routing
            %Similar to VLookup in excel, finds the true statement and returns the 1
            %column of the selected row
            
            channel = zeros(nchannels,1);
            
            for i=1:nchannels
                
                channel(i,1) = AVIL_channel_map((AVIL_channel_map(:,2) == placementArray(h).placement(i,2) & AVIL_channel_map(:,3) == placementArray(h).placement(i,3)),1);
                
                avil_file(:,channel(i,1)) = new_mix(:,i);
            end
            %% Rendering file
            folder = 'inAudio__64chn__downmixes\';
            newfilename = ([folder '__' num2str(c) '__64chn__' char(dmixFormat(dmixIdx)) '__' char(stimulusName{stimulusIdx}) '.wav'])
            audiowrite(newfilename,avil_file,Fs);
            c=c+1;
            
        end
    end
end


end