function [outputArg1] = formatdisp(placementArray)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here



titles={'Reference','Angle Distortion','Distance Distortion','Combined Distortion'};
smallmarker=8;
bigmarker=14;
figure(1)
for k=2:1:5
    subplot(2,2,k-1)
    set(gcf,'color','w');
    set(gca,'fontsize', 18);
    for i=1:2
        theta = deg2rad(placementArray(k).placement(i,2));
        phi = deg2rad(placementArray(k).placement(i,3));
        rho = placementArray(k).placement(i,4)+1;
        p=polarplot(theta,rho,'o')
        p.Color='b';
        p.Marker ='o';
        p.MarkerSize = smallmarker;
        rlim([0 4])
        rticks([2 3 4])
        rticklabels({'r = 2','r = 3','r = 4'})
        title([num2str(size(placementArray(k).placement,1)) 'chn ' char(titles(k-1)) ' Layout'] )
        hold on
        list=polarplot(0,0)
        list.Color='r'
        list.Marker ='square'
        list.MarkerSize = smallmarker;
    end
    

end
qw{1} = polarplot(nan, 'b o');
qw{2} = polarplot(nan, 'r square'); % You can add an extra element to
legend([qw{:}], {'Speakers 0° Elev.','Listener'}, 'location', 'best')
%% 5 channel format
figure(2)

for k=7:1:10
    subplot(2,2,k-6)
    set(gcf,'color','w');
    set(gca,'fontsize', 18);
    for i=1:5
        theta = deg2rad(placementArray(k).placement(i,2));
        phi = deg2rad(placementArray(k).placement(i,3));
        rho = placementArray(k).placement(i,4)+1;
        p=polarplot(theta,rho,'o');
        p.Color='b';
        p.Marker ='o';
        p.MarkerSize = smallmarker;
        rlim([0 4])
        rticks([2 3 4])
        rticklabels({'r = 2','r = 3','r = 4'})
        title([num2str(size(placementArray(k).placement,1)) 'chn ' char(titles(k-6)) ' Layout'] )
        hold on
        list=polarplot(0,0);
        list.Marker ='square';
        list.Color='r'
        list.MarkerSize =smallmarker;
    end
end
qw{1} = polarplot(nan, 'b o');
qw{2} = polarplot(nan, 'r square'); % You can add an extra element to
legend([qw{:}], {'Speakers 0° Elev.','Listener'}, 'location', 'best')

%% 12 channel formats
figure(3)
set(gcf,'color','w');
set(gca,'fontsize', 18);
for k=12:1:15
    list=polarplot(0,0)
    list.Color='r'
    list.Marker ='square'
    list.MarkerSize = 8;
        subplot(2,2,k-11)
          set(gcf,'color','w');
    set(gca,'fontsize', 18);
        for i=1:12
            if i~=4 % taking the LFE from the format
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
                p.Color='g'
                p.Marker ='o'
                p.MarkerSize = bigmarker;
            end
            
            rlim([0 4])
        rticks([2 3 4])
        rticklabels({'r = 2','r = 3','r = 4'})
            title([num2str(size(placementArray(k).placement,1)) 'chn ' char(titles(k-11)) ' Layout'] )
            hold on
            end
                       
        end
end
qw{1} = polarplot(nan, 'b o');
p.MarkerSize = smallmarker;
qw{2} = polarplot(nan, 'g o');
p.MarkerSize = bigmarker;
qw{3} = polarplot(nan, 'r square'); % You can add an extra element to
legend([qw{:}], {'Speakers 0° Elev.','Speakers 28° Elev.','Listener'}, 'location', 'best')
outputArg1 = titles;
end






