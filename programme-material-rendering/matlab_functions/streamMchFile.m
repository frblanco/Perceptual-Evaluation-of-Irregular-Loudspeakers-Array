function [ output_args ] = streamMchFile( inFile, scale, chanMap, bufSize, devName, outLimit)
%
%
% Use Matlab audioplayer framework to stream multichannel file from disk

% Note: Matlab must be set to use the ASIO drivers for the DSP toolbox in
% Preferences / DSP Toolbox

% Initialize player
player = audioDeviceWriter('Driver','ASIO',...
                         'DeviceName',devName,...
                         'SampleRate', 48000,...
                         'BitDepth','24-bit integer',...
                         'BufferSize', bufSize,...
                         'ChannelMappingSource','Property',...
                         'ChannelMapping',chanMap);
if ~iscell(inFile)
    inFile = {inFile};
end

if ~isscalar(scale)
    lsScale = scale(1);
    subScale = scale(2);
else
    lsScale = scale;
    subScale = 1;
end

if isempty(outLimit)
    outLimit = 0.3;
end

for ii = 1:length(inFile)
    % create file reader object
    reader = dsp.AudioFileReader('Filename',inFile{ii});
    reader.SamplesPerFrame = player.BufferSize;
    fprintf('\n Playing %s\n',inFile{ii});
    
    % check number of channels
    nChan = reader.info.NumChannels;
    doTrunc = 0;
    doSubMix = 0;
    if nChan > length(chanMap)
        doTrunc = 1;
        fprintf(' Fewer output channels specified than in file. Using first %g channels.\n', length(chanMap));
    elseif nChan < length(chanMap)
        doSubMix = 1;
        nExtraChan = length(chanMap)-nChan;
        fprintf(' More output channels specified than in file. Placing summed signal on extra channels.\n');
    end
    fprintf('      ');
    % stream to audio device
    sCount = 0;
    while ~isDone(reader)
        audioData = lsScale.*step(reader);
        if doTrunc
            audioData = audioData(:,1:length(chanMap));
        end
        if doSubMix
            audioData = [audioData repmat(sum(audioData,2).*subScale./sqrt(nChan),1,nExtraChan)];
        end
        if max(rms(audioData)) > outLimit
            fprintf('\n');
            warning('Audio output level too high, playback stopped. Try a lower scaling.');
            break;
        end
        step(player,audioData);
        sCount = sCount + 1;
        fprintf('\b\b\b\b\b\b%6.2f',sCount*player.BufferSize/player.SampleRate)
    end
    
end
    release(reader);
    release(player);
    fprintf('\n Done.\n');
end

