clc
clear all
close all


%addpath('inAudio__downmixes\12ch_dmw-final')
inAudioPath = 'inAudio__downmixes\12ch_dmw-final\*.wav';
s=dir(inAudioPath);
mov_win = 100;
fftsize = 2048;
titles = {'Excerpt 1 - 2chn','Excerpt 2 - 2chn','Excerpt 3 - 2chn','Excerpt 4 - 2chn','Excerpt 1 - 5chn','Excerpt 2 - 5chn','Excerpt 3 - 5chn','Excerpt 4 - 5chn','Excerpt 1 - 12chn','Excerpt 2 - 12chn','Excerpt 3 - 12chn','Excerpt 4 - 12chn'};
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
    E(i,2)=(mean(cell2mat(f_chn(i))))./(mean(cell2mat(b_chn(i))));%front/back
    E(i,3)=(mean(cell2mat(l_chn(i))))./(mean(cell2mat(r_chn(i))));%left/right
   
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
    %Periodgram'
 
    for i=1:4
    figure(1)
     subplot(2,2,i)
     set(gcf,'color','w');
    set(gca,'fontsize', 14);
    N =length(cell2mat(l_chn(i)));
    window = hann(N);
    [pxxl,f] = periodogram((cell2mat(l_chn(i))),window,fftsize,fs);
    [pxxr,f] = periodogram((cell2mat(r_chn(i))),window,fftsize,fs);
    plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r')
    xlabel('Hz')
    ylabel('dB/Hz')
    title(['Periodogram of ' titles{i}])
    legend('Left zone','Right Zone')
    end
  for i=5:8
      
        N =length(cell2mat(l_chn(i)));
        window = hann(N);
        figure(2)
           subplot(2,2,i-4)
        set(gcf,'color','w');
    set(gca,'fontsize', 14);
        [pxxl,f] = periodogram((cell2mat(l_chn(i))),window,fftsize,fs);
        [pxxr,f] = periodogram((cell2mat(r_chn(i))),window,fftsize,fs);
        plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r')
        xlabel('Hz')
        ylabel('dB/Hz')
        title(['Periodogram of ' titles{i}])
        legend('Left zone','Right Zone')
        figure(3)
        subplot(2,2,i-4) 
        set(gcf,'color','w');
        set(gca,'fontsize', 14);
        [pxxb,f] = periodogram(((rms(cell2mat(b_chn(i)),2))),window,fftsize,fs);
        [pxxf,f] = periodogram(((rms(cell2mat(f_chn(i)),2))),window,fftsize,fs);
        plot(f,movmean(10*log10(pxxb),mov_win),'g',f,movmean(10*log10(pxxf),mov_win),'r')
        xlabel('Hz')
        ylabel('dB/Hz')
        title(['Periodogram of ' titles{i}])
        legend('Front Zone','Rear Zone')
      end




  for i=9:12
        
        N =length(cell2mat(l_chn(i)));
        window = hann(N);
        figure(4)
        subplot(2,2,i-8)        
        set(gcf,'color','w');
        set(gca,'fontsize', 14);
        [pxxl,f] = periodogram((cell2mat(l_chn(i))),window,fftsize,fs);
        [pxxr,f] = periodogram((cell2mat(r_chn(i))),window,fftsize,fs);
        plot(f,movmean(10*log10(pxxl),mov_win),'g',f,movmean(10*log10(pxxr),mov_win),'r')
        xlabel('Hz')
        ylabel('dB/Hz')
        title(['Periodogram of ' titles{i}])
        legend('Left zone','Right Zone')
        figure(5)
        subplot(2,2,i-8) 
        set(gcf,'color','w');
        set(gca,'fontsize', 14);
        [pxxb,f] = periodogram(((rms(cell2mat(b_chn(i)),2))),window,fftsize,fs);
        [pxxf,f] = periodogram(((rms(cell2mat(f_chn(i)),2))),window,fftsize,fs);
        plot(f,movmean(10*log10(pxxb),mov_win),'g',f,movmean(10*log10(pxxf),mov_win),'r')
        xlabel('Hz')
        ylabel('dB/Hz')
        title(['Periodogram of ' titles{i}])
        legend('Front Zone','Rear Zone')
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



