function [outputArg1] = formatdisp(placementArray)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
set(gcf,'color','w');
titles={'Reference','Angle Distortion','Distance Distortion','Combined Distortion'};
smallmarker=8;
bigmarker=14;
figure(1)
for k=2:1:5
    subplot(2,2,k-1)
    for i=1:2
        theta = deg2rad(placementArray(k).placement(i,2));
        phi = deg2rad(placementArray(k).placement(i,3));
        rho = placementArray(k).placement(i,4)+1;
        p=polarplot(theta,rho,'o')
        p.Color='b';
        p.Marker ='o';
        p.MarkerSize = smallmarker;
        rlim([0 4])
        rticks([1 2  3])
        rticklabels({'r = 1','r = 2','r = 3'})
        title([num2str(size(placementArray(k).placement,1)) 'Channels ' char(titles(k-1)) ' Layout'] )
        hold on
        list=polarplot(0,0)
        list.Color='r'
        list.Marker ='square'
        list.MarkerSize = smallmarker;
    end
end

%% 5 channel format
set(gcf,'color','w');
figure(2)
for k=7:1:10
    subplot(2,2,k-6)
    for i=1:5
        theta = deg2rad(placementArray(k).placement(i,2));
        phi = deg2rad(placementArray(k).placement(i,3));
        rho = placementArray(k).placement(i,4)+1;
        p=polarplot(theta,rho,'o');
        p.Color='b';
        p.Marker ='o';
        p.MarkerSize = smallmarker;
        rlim([0 4])
        rticks([1 2  3])
        rticklabels({'r = 1','r = 2','r = 3'})
        title([num2str(size(placementArray(k).placement,1)) 'Channels ' char(titles(k-6)) ' Layout'] )
        hold on
        list=polarplot(0,0);
        list.Marker ='square';
        list.Color='r'
        list.MarkerSize =smallmarker;
    end
end


%% 12 channel formats
set(gcf,'color','w');
figure(3)
for k=12:1:15
    list=polarplot(0,0)
    list.Color='r'
    list.Marker ='square'
    list.MarkerSize = 8;
        subplot(2,2,k-11)
        for i=1:12
            theta = deg2rad(placementArray(k).placement(i,2));
            phi = deg2rad(placementArray(k).placement(i,3));
            rho = placementArray(k).placement(i,4)+1
            if phi == 0
                p=polarplot(theta,rho,'o')
                p.Color='b'
                p.Marker ='o'
                p.MarkerSize = smallmarker;
            else
                p=polarplot(theta,rho,'o')
                p.Color='b'
                p.Marker ='o'
                p.MarkerSize = bigmarker;
            end
            
            rlim([0 4])
            rticks([1 2 3])
            rticklabels({'r = 1','r = 2','r = 3'})
            title([num2str(size(placementArray(k).placement,1)) 'Channels ' char(titles(k-11)) ' Layout'] )
            hold on
            
            
            
        end
end
outputArg1 = titles;
end






