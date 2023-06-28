clc
clear all
close all


%addpath('inAudio__downmixes\12ch_dmw-final')
%inAudioPath = 'inAudio__downmixes\12ch_dmw-final\*.wav';
inAudioPath = 'Program_material_selection\*.wav';
s=dir(inAudioPath);
mov_win = 50;
fftsize = 2048;
for i=1:length(s)
    Ename(i,1) =string(s(i).name);
    [data,fs]=audioread([s(i).folder '\' s(i).name]);
    if size(data,2) == 2
        f_chn{i,1} = 0;
        b_chn{i,1} = 0;
        r_chn{i,1} = rms(data(:,2),2);
        l_chn{i,1} = rms(data(:,1),2);
    elseif size(data,2) == 5
        f_chn{i,1} =rms([data(:,2)  data(:,1) data(:,3)],2);
        b_chn{i,1} =rms([data(:,4) data(:,5)],2);
        r_chn{i,1} =rms([data(:,2) data(:,3) data(:,5)],2);
        l_chn{i,1} =rms([data(:,1) data(:,3) data(:,4)],2);
    elseif size(data,2) == 12
        f_chn{i,1} =rms([data(:,2)  data(:,1) data(:,3) data(:,9) data(:,10) ],2);
        b_chn{i,1} =rms([data(:,5) data(:,6) data(:,7) data(:,8) data(:,11) data(:,12)],2);
        r_chn{i,1} =rms([data(:,2) data(:,3) data(:,6) data(:,8) data(:,10) data(:,12)],2);
        l_chn{i,1} =rms([data(:,1) data(:,3) data(:,5) data(:,7) data(:,9) data(:,11)],2);
    end
    
    %rms
    E(i,2)= rms(cell2mat(f_chn(i)))/rms(cell2mat(b_chn(i)));%front/back
    E(i,3)=rms(cell2mat(l_chn(i)))/rms(cell2mat(r_chn(i)));%left/right
   
    %cross correlation
    [c,lags] = xcorr(cell2mat(f_chn(i)),cell2mat(b_chn(i)),'normalized');%front/back
    E(i,4) = max(c);
    [c,lags] = xcorr(cell2mat(l_chn(i)),cell2mat(r_chn(i)),'normalized');%left/right
    E(i,5) = max(c);
    
    if cell2mat(b_chn(i)) == 0
        E(i,2)=0;
    elseif cell2mat(r_chn(i)) == 0
        E(i,3)=0;
    end
    
end
    

 %% Periodgram'
titles = {'Excerpt 1 - 2chn','Excerpt 2 - 2chn','Excerpt 3 - 2chn','Excerpt 4 - 2chn','Excerpt 1 - 5chn','Excerpt 2 - 5chn','Excerpt 3 - 5chn','Excerpt 4 - 5chn','Excerpt 1 - 12chn','Excerpt 2 - 12chn','Excerpt 3 - 12chn','Excerpt 4 - 12chn'};
excerpt_idx = [1 5 9, 
			  2 6 10,
			  3 7 11,
			  4 8 12];
 N =length(cell2mat(l_chn(i)));
    window = hann(N);
for j=1:1:4
figure(j)
subplot(3,2,1)
%2channel right vs left  - 
 N =length(cell2mat(l_chn(excerpt_idx(j,1))));
 window = hann(N);
[pxxl,f] = periodogram((cell2mat(l_chn(excerpt_idx(j,1),1))),window,fftsize,fs);
[pxxr,f] = periodogram((cell2mat(r_chn(excerpt_idx(j,1),1))),window,fftsize,fs);
plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r');
ylim([-120 -60])
set(gcf,'color','w');
set(gca,'fontsize', 14);
xlabel('Hz')
ylabel('dB/Hz')
 title(['2 chn - Excerpt ' num2str(j)])
legend('Left zone','Right Zone')

subplot(3,2,3)
%5 channel right vs left -
 N =length(cell2mat(l_chn(excerpt_idx(j,2))));
 window = hann(N);
[pxxl,f] = periodogram((cell2mat(l_chn(excerpt_idx(j,2),1))),window,fftsize,fs);
[pxxr,f] = periodogram((cell2mat(r_chn(excerpt_idx(j,2),1))),window,fftsize,fs);
plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r');
ylim([-120 -60]);
set(gcf,'color','w');
set(gca,'fontsize', 14);
xlabel('Hz')
ylabel('dB/Hz')
 title(['5 chn - Excerpt ' num2str(j)])
legend('Left zone','Right Zone')


subplot(3,2,5.5)
%12 channel right vs left -
 N =length(cell2mat(l_chn(excerpt_idx(j,3))));
 window = hann(N);
[pxxl,f] = periodogram((cell2mat(l_chn(excerpt_idx(j,3),1))),window,fftsize,fs);
[pxxr,f] = periodogram((cell2mat(r_chn(excerpt_idx(j,3),1))),window,fftsize,fs);
plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r');
ylim([-120 -60])
set(gcf,'color','w');
set(gca,'fontsize', 14);
xlabel('Hz')
ylabel('dB/Hz')
 title(['12 chn - Excerpt ' num2str(j)])
legend('Left zone','Right Zone')

subplot(3,2,2)
%5 channel front vs rear -
N =length(cell2mat(b_chn(excerpt_idx(j,2))));
 window = hann(N);
[pxxb,f] = periodogram((cell2mat(b_chn(excerpt_idx(j,2),1))),window,fftsize,fs);
[pxxf,f] = periodogram((cell2mat(f_chn(excerpt_idx(j,2),1))),window,fftsize,fs);
plot(f,movmean(10*log10(pxxb),mov_win),'g',f,movmean(10*log10(pxxf),mov_win),'r');
ylim([-120 -60])
set(gcf,'color','w');
set(gca,'fontsize', 14);
xlabel('Hz')
ylabel('dB/Hz')
 title(['12 chn - Excerpt ' num2str(j)])
legend('Rear zone',' Rear Zone')



subplot(3,2,4)
%12 channel front vs rear - 
N =length(cell2mat(b_chn(excerpt_idx(j,3))));
 window = hann(N);
[pxxb,f] = periodogram((cell2mat(b_chn(excerpt_idx(j,3),1))),window,fftsize,fs);
[pxxf,f] = periodogram((cell2mat(f_chn(excerpt_idx(j,3),1))),window,fftsize,fs);
plot(f,movmean(10*log10(pxxb),mov_win),'g',f,movmean(10*log10(pxxf),mov_win),'r');
ylim([-120 -60])
set(gcf,'color','w');
set(gca,'fontsize', 14);
xlabel('Hz')
ylabel('dB/Hz')
 title(['12 chn - Excerpt ' num2str(j)])
legend('Rear zone',' Rear Zone')
end

        
%% Spectrogram
% 
 titles ={'2chn Excerpt 1','2chn Excerpt 2','2chn Excerpt 3','2chn Excerpt 4','5chn Excerpt 1','5chn Excerpt 2','5chn Excerpt 3','5chn Excerpt 4','12chn Excerpt 1','12chn Excerpt 2','12chn Excerpt 3','12chn Excerpt 4'};
% inAudioPath = 'inAudio__downmixes\12ch_dmw-final\*.wav';
% s=dir(inAudioPath);
figure(6)
 
for i=09:12
    subplot(2,2,(i-8))
    [data,fs]=audioread([s(i).folder '\' s(i).name]);
    spectrogram(rms(data,2),98,80,100,fs,'yaxis');
    set(gcf,'color','w');
    set(gca,'fontsize', 14);
    title(titles{i});
end



